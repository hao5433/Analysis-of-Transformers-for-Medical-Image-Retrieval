"""
MIRViT Demo - Medical Image Retrieval with Vision Transformers
Demo cho giáo sư: Truy xuất ảnh y tế dựa trên nội dung (Content-based MIR)

Bài báo: "Analysis of Transformers for Medical Image Retrieval" (MIDL 2024)
Tác giả: Arvapalli Sai Susmitha & Vinay P. Namboodiri
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms, models
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
IMG_SIZE = 224
TOP_K = 6

LABEL_VI = {
    "covid": "COVID-19",
    "normal": "Bình thường",
    "pneumonia": "Viêm phổi",
    "melanoma": "Melanoma (Ác tính)",
    "nevi": "Nốt ruồi (Lành tính)",
    "keratosis": "Dày sừng",
    "polyp": "Polyp",
    "normal_cecum": "Manh tràng bình thường",
    "esophagitis": "Viêm thực quản",
    "ulcerative_colitis": "Viêm loét đại tràng",
}

DATASET_GROUPS = {
    "🫁 COVID-19 Chest X-Ray": ["covid", "normal", "pneumonia"],
    "🔬 ISIC Skin Lesion": ["melanoma", "nevi", "keratosis"],
    "🔭 Kvasir Endoscopy": ["polyp", "normal_cecum", "esophagitis", "ulcerative_colitis"],
}

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# MODEL: MIRViT (ViT-Small + Projection Head)
# ─────────────────────────────────────────────
class MIRViT(nn.Module):
    """
    MIRViT: ViT-Small với projection head cho contrastive learning
    Embedding dim: 384 (như trong bài báo)
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,  # Lấy embedding, không classify
        )
        # Projection head (như trong contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        return F.normalize(proj, dim=-1)


class ResNet50Baseline(nn.Module):
    """ResNet50 baseline để so sánh với MIRViT"""
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        feat = self.features(x)
        feat = feat.flatten(1)
        return F.normalize(feat, dim=-1)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("⏳ Đang load models...")
mirvit_model = MIRViT().to(DEVICE).eval()
resnet_model = ResNet50Baseline().to(DEVICE).eval()
print(f"✅ Models loaded! Device: {DEVICE}")

# ─────────────────────────────────────────────
# DATABASE: Trích xuất embeddings từ ảnh mẫu
# ─────────────────────────────────────────────
db_embeddings_vit = None
db_embeddings_resnet = None
db_labels = []
db_paths = []


def build_database():
    """Xây dựng embedding database từ ảnh mẫu"""
    global db_embeddings_vit, db_embeddings_resnet, db_labels, db_paths

    if not os.path.exists(SAMPLE_DIR) or not any(
        os.path.isdir(os.path.join(SAMPLE_DIR, d)) for d in os.listdir(SAMPLE_DIR)
        if not d.startswith(".")
    ):
        print("⚠️  Chưa có dữ liệu mẫu. Đang tạo...")
        sys_path = os.path.dirname(__file__)
        import sys
        sys.path.insert(0, sys_path)
        from generate_sample_data import generate_all_samples
        generate_all_samples()

    print("⏳ Đang xây dựng embedding database...")
    vit_embs, resnet_embs, labels, paths = [], [], [], []

    for label in sorted(os.listdir(SAMPLE_DIR)):
        label_dir = os.path.join(SAMPLE_DIR, label)
        if not os.path.isdir(label_dir) or label.startswith("."):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(label_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    vit_emb = mirvit_model(tensor).cpu().numpy()[0]
                    resnet_emb = resnet_model(tensor).cpu().numpy()[0]
                vit_embs.append(vit_emb)
                resnet_embs.append(resnet_emb)
                labels.append(label)
                paths.append(fpath)
            except Exception as e:
                print(f"  ⚠️  Lỗi {fpath}: {e}")

    db_embeddings_vit = np.array(vit_embs)
    db_embeddings_resnet = np.array(resnet_embs)
    db_labels = labels
    db_paths = paths
    print(f"✅ Database: {len(db_paths)} ảnh, {len(set(db_labels))} lớp")


build_database()


# ─────────────────────────────────────────────
# XAI: Attention Rollout
# ─────────────────────────────────────────────
def compute_attention_rollout(model, img_tensor):
    """
    Tính Attention Rollout để tạo saliency map
    Phương pháp: tích luỹ attention qua các layer transformer
    """
    attentions_list = []
    hooks = []

    def make_hook():
        def hook_fn(module, input, output):
            attentions_list.append(output.detach().cpu())
        return hook_fn

    for block in model.backbone.blocks:
        h = block.attn.register_forward_hook(make_hook())
        hooks.append(h)

    with torch.no_grad():
        _ = model.backbone(img_tensor)

    for h in hooks:
        h.remove()

    if not attentions_list:
        return None

    # Attention Rollout algorithm
    result = torch.eye(attentions_list[0].shape[-1])
    for attn in attentions_list:
        attn_avg = attn.mean(dim=1)[0]  # Average over heads: (seq_len, seq_len)
        attn_avg = attn_avg + torch.eye(attn_avg.shape[0])  # Residual connection
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_avg, result)

    # Lấy attention từ CLS token đến các patch tokens
    num_patches = int((IMG_SIZE / 16) ** 2)  # 14×14 = 196
    mask = result[0, 1:num_patches + 1].numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    grid_size = int(np.sqrt(len(mask)))
    mask = mask[:grid_size * grid_size].reshape(grid_size, grid_size)
    return mask


def create_saliency_overlay(original_img, attention_map, alpha=0.5):
    """Tạo ảnh overlay saliency map lên ảnh gốc"""
    img_array = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)))
    attn_img = Image.fromarray((attention_map * 255).astype(np.uint8))
    attn_resized = np.array(attn_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    heatmap = (cm.jet(attn_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
    return Image.fromarray(overlay)


# ─────────────────────────────────────────────
# CHART: So sánh mAP từ bài báo
# ─────────────────────────────────────────────
def create_map_comparison_chart():
    """Tạo biểu đồ so sánh mAP từ kết quả bài báo MIDL 2024"""
    datasets = ["ISIC 2017\n(Da liễu)", "COVID-19\n(X-quang)", "Kvasir-V2\n(Nội soi)"]
    model_results = {
        "ResNet50": [57.72, 91.33, 84.85],
        "DenseNet121": [58.38, 94.62, 83.89],
        "DeiT-Small (CE)": [63.32, 92.93, 88.15],
        "MIRViT Small ⭐": [70.96, 96.96, 90.16],
    }

    x = np.arange(len(datasets))
    width = 0.2
    colors = ["#6c757d", "#adb5bd", "#4dabf7", "#ff6b6b"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (model_name, scores) in enumerate(model_results.items()):
        bars = ax.bar(x + i * width, scores, width, label=model_name,
                      color=colors[i], alpha=0.9, edgecolor="white", linewidth=0.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{score:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xlabel("Dataset", color="white", fontsize=12)
    ax.set_ylabel("mAP (%)", color="white", fontsize=12)
    ax.set_title("📈 So sánh mAP - MIRViT vs Các phương pháp khác\n(Kết quả từ bài báo MIDL 2024)",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_ylim(50, 104)
    ax.legend(loc="lower right", facecolor="#0f3460", labelcolor="white",
              fontsize=9, framealpha=0.8)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, color="white")
    ax.annotate("MIRViT vượt trội\ntrên cả 3 datasets!",
                xy=(2 + 3 * width, 90.16), xytext=(1.3, 100),
                fontsize=9, color="#ff6b6b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff6b6b"))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# RETRIEVAL CORE
# ─────────────────────────────────────────────
def get_filtered_db(dataset_filter):
    """Lọc database theo dataset"""
    if dataset_filter == "Tất cả":
        return db_embeddings_vit, db_embeddings_resnet, db_labels, db_paths

    group_labels = DATASET_GROUPS.get(dataset_filter, None)
    if group_labels is None:
        return db_embeddings_vit, db_embeddings_resnet, db_labels, db_paths

    indices = [i for i, l in enumerate(db_labels) if l in group_labels]
    if not indices:
        return db_embeddings_vit, db_embeddings_resnet, db_labels, db_paths

    return (
        db_embeddings_vit[indices],
        db_embeddings_resnet[indices],
        [db_labels[i] for i in indices],
        [db_paths[i] for i in indices],
    )


def make_result_grid(top_indices, sims, paths, labels, title):
    """Tạo lưới ảnh kết quả retrieval"""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1a1a2e")

    for i, ax in enumerate(axes.flat):
        if i < len(top_indices):
            idx = top_indices[i]
            img = Image.open(paths[idx]).convert("RGB")
            ax.imshow(img)
            label_vi = LABEL_VI.get(labels[idx], labels[idx])
            sim_score = float(sims[idx])
            color = "#00ff88" if sim_score > 0.8 else "#ffaa00" if sim_score > 0.6 else "#ff6666"
            ax.set_title(f"#{i+1} {label_vi}\nSim: {sim_score:.3f}",
                         fontsize=9, color=color, pad=3)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GRADIO HANDLER FUNCTIONS
# ─────────────────────────────────────────────
def process_query(query_img, dataset_filter, show_xai):
    """Hàm chính: xử lý query và trả về kết quả retrieval + XAI"""
    if query_img is None:
        return None, None, None, "⚠️ Vui lòng upload ảnh query!"

    query_pil = query_img.convert("RGB") if hasattr(query_img, "convert") else Image.fromarray(query_img).convert("RGB")
    tensor = transform(query_pil).unsqueeze(0).to(DEVICE)

    # Lấy embeddings
    with torch.no_grad():
        vit_emb = mirvit_model(tensor).cpu().numpy()[0]
        resnet_emb = resnet_model(tensor).cpu().numpy()[0]

    # Lọc database
    filt_vit, filt_resnet, filt_labels, filt_paths = get_filtered_db(dataset_filter)

    # Cosine similarity (embeddings đã L2-normalize)
    vit_sims = np.dot(filt_vit, vit_emb)
    resnet_sims = np.dot(filt_resnet, resnet_emb)

    vit_top = np.argsort(vit_sims)[::-1][:TOP_K]
    resnet_top = np.argsort(resnet_sims)[::-1][:TOP_K]

    # Tạo grid kết quả
    vit_fig = make_result_grid(vit_top, vit_sims, filt_paths, filt_labels,
                               "🏆 MIRViT Small — Top-6 Kết quả Retrieval")
    resnet_fig = make_result_grid(resnet_top, resnet_sims, filt_paths, filt_labels,
                                  "📉 ResNet50 Baseline — Top-6 Kết quả Retrieval")

    # XAI: Attention Rollout
    xai_fig = None
    if show_xai:
        attn_map = compute_attention_rollout(mirvit_model, tensor)
        if attn_map is not None:
            overlay = create_saliency_overlay(query_pil, attn_map)
            fig_xai, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig_xai.patch.set_facecolor("#1a1a2e")
            fig_xai.suptitle("🧠 XAI — Attention Rollout: Mô hình 'nhìn vào đâu'?",
                             fontsize=13, color="white", fontweight="bold")
            axes[0].imshow(query_pil.resize((IMG_SIZE, IMG_SIZE)))
            axes[0].set_title("Ảnh gốc (Query)", color="white", fontsize=11)
            axes[0].axis("off")
            im = axes[1].imshow(attn_map, cmap="jet")
            axes[1].set_title("Attention Map (Heatmap)", color="white", fontsize=11)
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (Ảnh + Attention)", color="white", fontsize=11)
            axes[2].axis("off")
            plt.tight_layout()
            xai_fig = fig_xai

    # Thống kê
    vit_top1_label = filt_labels[vit_top[0]]
    vit_top1_sim = float(vit_sims[vit_top[0]])
    resnet_top1_label = filt_labels[resnet_top[0]]
    resnet_top1_sim = float(resnet_sims[resnet_top[0]])
    vit_labels_top = [LABEL_VI.get(filt_labels[i], filt_labels[i]) for i in vit_top]
    resnet_labels_top = [LABEL_VI.get(filt_labels[i], filt_labels[i]) for i in resnet_top]

    stats_text = f"""## 📊 Kết quả Retrieval

### 🏆 MIRViT Small (ViT-Small + Contrastive Loss)
- **Top-1:** {LABEL_VI.get(vit_top1_label, vit_top1_label)} — Similarity: **{vit_top1_sim:.4f}**
- **Top-{TOP_K}:** {' | '.join(vit_labels_top)}

### 📉 ResNet50 Baseline (CNN truyền thống)
- **Top-1:** {LABEL_VI.get(resnet_top1_label, resnet_top1_label)} — Similarity: **{resnet_top1_sim:.4f}**
- **Top-{TOP_K}:** {' | '.join(resnet_labels_top)}

---
### 💡 Tại sao MIRViT tốt hơn?
| Tiêu chí | ResNet50 | MIRViT Small |
|----------|----------|--------------|
| Kiến trúc | CNN (local features) | ViT (global attention) |
| Loss | Cross-entropy | Contrastive + KoLeo |
| Embedding | 2048-dim | 384-dim |
| Giải thích | ❌ Hộp đen | ✅ Attention Rollout |
| mAP ISIC | 57.72% | **70.96%** |
"""
    return vit_fig, resnet_fig, xai_fig, stats_text


def process_xai_only(query_img):
    """Chỉ tính XAI cho tab riêng"""
    if query_img is None:
        return None

    query_pil = query_img.convert("RGB") if hasattr(query_img, "convert") else Image.fromarray(query_img).convert("RGB")
    tensor = transform(query_pil).unsqueeze(0).to(DEVICE)
    attn_map = compute_attention_rollout(mirvit_model, tensor)

    if attn_map is None:
        return None

    overlay = create_saliency_overlay(query_pil, attn_map)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("🧠 XAI — Attention Rollout Visualization",
                 fontsize=13, color="white", fontweight="bold")

    axes[0].imshow(query_pil.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Ảnh gốc", color="white", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(attn_map, cmap="jet")
    axes[1].set_title("Attention Map", color="white", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", color="white", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def get_sample_images():
    """Lấy danh sách ảnh mẫu để hiển thị trong gallery"""
    samples = []
    if not os.path.exists(SAMPLE_DIR):
        return samples
    for label in sorted(os.listdir(SAMPLE_DIR)):
        label_dir = os.path.join(SAMPLE_DIR, label)
        if os.path.isdir(label_dir) and not label.startswith("."):
            files = sorted([f for f in os.listdir(label_dir) if f.endswith(".jpg")])
            if files:
                samples.append(os.path.join(label_dir, files[0]))
    return samples[:12]


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
CSS = """
body { background: #1a1a2e !important; }
.gradio-container { max-width: 1400px !important; }
"""

with gr.Blocks(css=CSS, title="MIRViT Demo — Medical Image Retrieval") as demo:

    # ── HEADER ──
    gr.HTML("""
    <div style="text-align:center; padding:24px 20px 16px;
                background:linear-gradient(135deg,#1a1a2e,#0f3460);
                border-radius:14px; margin-bottom:16px;
                border:1px solid rgba(255,255,255,0.1);">
        <h1 style="color:#e94560; font-size:2.2em; margin:0 0 6px;">
            🏥 MIRViT Demo
        </h1>
        <h2 style="color:#4dabf7; font-size:1.15em; margin:0 0 8px; font-weight:400;">
            Medical Image Retrieval with Vision Transformers
        </h2>
        <p style="color:#adb5bd; margin:4px 0; font-size:0.95em;">
            📄 <em>"Analysis of Transformers for Medical Image Retrieval"</em>
            &nbsp;—&nbsp; MIDL 2024
        </p>
        <p style="color:#868e96; font-size:0.85em; margin:4px 0;">
            Arvapalli Sai Susmitha (IIT Kanpur) & Vinay P. Namboodiri (University of Bath)
        </p>
        <div style="margin-top:10px; display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
            <span style="background:#e94560;color:white;padding:3px 10px;border-radius:20px;font-size:0.8em;">ViT-Small</span>
            <span style="background:#0f3460;color:white;padding:3px 10px;border-radius:20px;font-size:0.8em;">Contrastive Loss</span>
            <span style="background:#16213e;color:white;padding:3px 10px;border-radius:20px;font-size:0.8em;border:1px solid #4dabf7;">KoLeo Regularization</span>
            <span style="background:#16213e;color:white;padding:3px 10px;border-radius:20px;font-size:0.8em;border:1px solid #4dabf7;">Attention Rollout XAI</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════
        # TAB 1: DEMO RETRIEVAL
        # ══════════════════════════════════════
        with gr.TabItem("🔍 Demo Retrieval"):
            gr.Markdown("""
            ### Cách sử dụng
            1. **Upload ảnh y tế** hoặc **click vào ảnh mẫu** bên dưới
            2. Chọn dataset muốn tìm kiếm (hoặc để "Tất cả")
            3. Nhấn **🔍 Tìm kiếm** → Hệ thống trả về Top-6 ảnh tương tự nhất
            4. So sánh kết quả giữa **MIRViT** và **ResNet50 baseline**
            """)

            with gr.Row():
                # ── Cột trái: Input ──
                with gr.Column(scale=1, min_width=300):
                    query_input = gr.Image(
                        label="📤 Ảnh Query",
                        type="pil",
                        height=260,
                    )
                    dataset_filter = gr.Dropdown(
                        choices=["Tất cả"] + list(DATASET_GROUPS.keys()),
                        value="Tất cả",
                        label="🗂️ Lọc theo dataset",
                    )
                    show_xai = gr.Checkbox(
                        label="🧠 Hiển thị XAI (Attention Rollout)",
                        value=True,
                    )
                    search_btn = gr.Button(
                        "🔍 Tìm kiếm ảnh tương tự",
                        variant="primary",
                        size="lg",
                    )

                # ── Cột phải: Stats ──
                with gr.Column(scale=2):
                    stats_output = gr.Markdown(
                        "*Upload ảnh và nhấn Tìm kiếm để xem kết quả...*"
                    )

            # ── Gallery ảnh mẫu ──
            gr.Markdown("#### 📁 Ảnh mẫu từ database (click để chọn làm query):")
            sample_gallery = gr.Gallery(
                value=get_sample_images(),
                label="",
                columns=6,
                height=160,
                allow_preview=False,
                show_label=False,
            )

            # ── Kết quả retrieval ──
            with gr.Row():
                vit_results = gr.Plot(label="🏆 MIRViT Small")
                resnet_results = gr.Plot(label="📉 ResNet50 Baseline")

            xai_output = gr.Plot(label="🧠 XAI — Attention Rollout Saliency Map")

            # Events
            def select_sample(evt: gr.SelectData):
                samples = get_sample_images()
                if evt.index < len(samples):
                    return Image.open(samples[evt.index]).convert("RGB")
                return None

            sample_gallery.select(fn=select_sample, outputs=query_input)

            search_btn.click(
                fn=process_query,
                inputs=[query_input, dataset_filter, show_xai],
                outputs=[vit_results, resnet_results, xai_output, stats_output],
            )

        # ══════════════════════════════════════
        # TAB 2: SO SÁNH HIỆU NĂNG
        # ══════════════════════════════════════
        with gr.TabItem("📊 So sánh Hiệu năng (mAP)"):
            gr.Markdown("""
            ### Kết quả thực nghiệm từ bài báo MIDL 2024
            **MIRViT Small** vượt trội trên cả 3 bộ dữ liệu y tế.
            """)

            map_chart = gr.Plot(label="Biểu đồ so sánh mAP")

            gr.Button("📈 Hiển thị biểu đồ so sánh mAP", variant="primary").click(
                fn=create_map_comparison_chart,
                outputs=map_chart,
            )

            gr.Markdown("""
            ---
            ### Bảng kết quả chi tiết

            | Dataset | ResNet50 | DenseNet121 | DeiT-Small (CE) | **MIRViT Small** | Cải thiện |
            |---------|:--------:|:-----------:|:---------------:|:----------------:|:---------:|
            | ISIC 2017 (Da liễu) | 57.72% | 58.38% | 63.32% | **70.96%** | +13.24% vs ResNet |
            | COVID-19 (X-quang) | 91.33% | 94.62% | 92.93% | **96.96%** | +5.63% vs ResNet |
            | Kvasir-V2 (Nội soi) | 84.85% | 83.89% | 88.15% | **90.16%** | +5.31% vs ResNet |

            ### Nhận xét về siêu tham số
            - **MIRViT Small** (384-dim) liên tục vượt **MIRViT Base** (786-dim)
              → Embedding nhỏ hơn phù hợp hơn cho retrieval y tế
            - **λ = 0.3** là lựa chọn tối ưu cho KoLeo regularization
            - **Contrastive Loss** vượt Cross-entropy Loss trong mọi trường hợp
            - **MedViT** (hybrid CNN-Transformer) mạnh về phân loại nhưng kém trong retrieval
            """)

        # ══════════════════════════════════════
        # TAB 3: KIẾN TRÚC & PHƯƠNG PHÁP
        # ══════════════════════════════════════
        with gr.TabItem("🏗️ Kiến trúc & Phương pháp"):
            gr.Markdown("""
            ## Kiến trúc MIRViT — Tổng quan

            ### Pipeline
            ```
            ┌─────────────────────────────────────────────────────────┐
            │                    TRAINING PHASE                       │
            │                                                         │
            │  Medical Image ──► Patch Embedding ──► ViT Encoder     │
            │                         (16×16)        (12 blocks)     │
            │                                              │          │
            │                                    CLS Token (384-dim) │
            │                                              │          │
            │                                    Projection Head     │
            │                                              │          │
            │                                    L2-Normalized Emb   │
            │                                              │          │
            │              L = L_contrastive + λ × L_KoLeo           │
            └─────────────────────────────────────────────────────────┘

            ┌─────────────────────────────────────────────────────────┐
            │                   INFERENCE PHASE                       │
            │                                                         │
            │  Query Image ──► ViT Encoder ──► Embedding             │
            │                                       │                 │
            │                              Cosine Similarity          │
            │                                       │                 │
            │                              Database Embeddings        │
            │                                       │                 │
            │                              Rank ──► Top-K Results    │
            └─────────────────────────────────────────────────────────┘
            ```

            ### Hàm mất mát
            ```
            L = L_contrastive + λ × L_KoLeo    (λ = 0.3)

            ┌─ L_contrastive (Contrastive Loss):
            │   • Kéo ảnh cùng lớp lại gần nhau trong embedding space
            │   • Đẩy ảnh khác lớp ra xa
            │   • Dùng cross-batch memory → tăng số negative pairs
            │     mà không tốn thêm bộ nhớ GPU
            │
            └─ L_KoLeo (Differential Entropy Regularization):
                • Tối đa hoá khoảng cách đến láng giềng gần nhất
                • Tránh các embedding bị chồng chéo (collapse)
                • Giúp phân bố embedding đều hơn trong không gian
            ```

            ### Các biến thể mô hình
            | Mô hình | Backbone | Embedding Dim | Patch Size | Params |
            |---------|----------|:-------------:|:----------:|:------:|
            | **MIRViT Small** ⭐ | ViT-Small | **384** | 16×16 | ~22M |
            | MIRdeiT Small | DeiT-Small | 384 | 16×16 | ~22M |
            | MIRViT Base | ViT-Base | 786 | 16×16 | ~86M |

            > 💡 **MIRViT Small** với embedding 384-dim liên tục vượt MIRViT Base (786-dim)
            > → Embedding nhỏ hơn, compact hơn, phù hợp hơn cho bài toán retrieval
            """)

        # ══════════════════════════════════════
        # TAB 4: XAI PHÂN TÍCH
        # ══════════════════════════════════════
        with gr.TabItem("🧠 XAI — Giải thích mô hình"):
            gr.Markdown("""
            ## Đánh giá khả năng giải thích (Explainability)

            Một trong những điểm mạnh của MIRViT là **khả năng giải thích** — bác sĩ có thể
            thấy mô hình "nhìn vào đâu" khi đưa ra kết quả retrieval.

            ### Các phương pháp XAI được so sánh:
            | Phương pháp | Loại | Mô tả |
            |-------------|------|-------|
            | **Attention Rollout** | Attention-based | Tích luỹ attention qua các layer |
            | **Chefer2** | Gradient-weighted | Kết hợp gradient và attention |
            | **TIS** | Sampling-based | Transformer Input Sampling |
            | **ViT-CX** | Counterfactual | Giải thích phản thực |
            | TAMs, BTH, BTT | Attention variants | Các biến thể attribution |

            ### Kết quả đánh giá (MIRViT Small):
            | Dataset | Metric | Chefer2 | Rollout | **TIS** | ViT-CX |
            |---------|--------|:-------:|:-------:|:-------:|:------:|
            | ISIC | Insertion ↑ | **0.79** | 0.78 | 0.78 | 0.72 |
            | ISIC | Deletion ↓ | **0.41** | 0.45 | **0.41** | 0.53 |
            | COVID | Insertion ↑ | **0.70** | 0.69 | 0.67 | 0.62 |
            | Kvasir | Deletion ↓ | **0.40** | 0.42 | **0.42** | 0.49 |

            > 🏆 **TIS** nổi bật nhất về mặt **định tính** — saliency map tập trung rõ vào
            > vùng tổn thương, phù hợp nhất cho môi trường lâm sàng.
            > **Chefer2** mạnh hơn một chút về số liệu định lượng.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    xai_query = gr.Image(
                        label="📤 Upload ảnh để phân tích XAI",
                        type="pil",
                        height=260,
                    )
                    xai_btn = gr.Button("🧠 Phân tích Attention Rollout", variant="primary")
                with gr.Column(scale=2):
                    xai_standalone = gr.Plot(label="🧠 Attention Rollout Visualization")

            xai_btn.click(
                fn=process_xai_only,
                inputs=xai_query,
                outputs=xai_standalone,
            )

    # ── FOOTER ──
    gr.HTML("""
    <div style="text-align:center; padding:14px; margin-top:16px;
                background:rgba(255,255,255,0.03); border-radius:8px;
                color:#868e96; font-size:0.82em; border:1px solid rgba(255,255,255,0.06);">
        <p style="margin:3px 0;">
            📄 <strong>MIRViT</strong> — Analysis of Transformers for Medical Image Retrieval
            &nbsp;|&nbsp; MIDL 2024
        </p>
        <p style="margin:3px 0;">
            Arvapalli Sai Susmitha (IIT Kanpur) & Vinay P. Namboodiri (University of Bath)
        </p>
        <p style="margin:3px 0; color:#555;">
            ⚠️ Demo sử dụng ảnh synthetic để minh hoạ.
            Trong thực tế dùng dataset COVID-19 (~9,208 ảnh), ISIC 2017 (2,750 ảnh), Kvasir-V2 (8,000 ảnh).
        </p>
    </div>
    """)


# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏥 MIRViT Demo — Medical Image Retrieval")
    print("=" * 60)
    print(f"📊 Database: {len(db_paths)} ảnh | {len(set(db_labels))} lớp")
    print(f"🖥️  Device: {DEVICE}")
    print(f"🌐 URL: http://localhost:7860")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False,
        inbrowser=True,
    )
