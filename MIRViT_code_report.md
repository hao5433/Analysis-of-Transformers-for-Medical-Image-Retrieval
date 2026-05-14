# Báo Cáo Chi Tiết: Demo MIRViT — Medical Image Retrieval

> **Bài báo gốc:** "Analysis of Transformers for Medical Image Retrieval" — MIDL 2024  
> **Tác giả:** Arvapalli Sai Susmitha (IIT Kanpur) & Vinay P. Namboodiri (University of Bath)  
> **Demo code:** `demo/app.py` + `demo/generate_sample_data.py`

---

## Mục lục

1. [Tổng quan demo](#1-tổng-quan-demo)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Giải thích code chi tiết](#3-giải-thích-code-chi-tiết)
   - [3.1 Cấu hình & hằng số](#31-cấu-hình--hằng-số)
   - [3.2 Mô hình MIRViT](#32-mô-hình-mirvit)
   - [3.3 Mô hình ResNet50 Baseline](#33-mô-hình-resnet50-baseline)
   - [3.4 Xây dựng Embedding Database](#34-xây-dựng-embedding-database)
   - [3.5 XAI — Attention Rollout](#35-xai--attention-rollout)
   - [3.6 Retrieval Core](#36-retrieval-core)
   - [3.7 Giao diện Gradio](#37-giao-diện-gradio)
   - [3.8 Tạo dữ liệu mẫu](#38-tạo-dữ-liệu-mẫu)
4. [So sánh code với bài báo](#4-so-sánh-code-với-bài-báo)
5. [Hướng dẫn chạy trên local](#5-hướng-dẫn-chạy-trên-local)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Tổng quan demo

Demo này **minh hoạ** hệ thống truy xuất ảnh y tế (Medical Image Retrieval — MIR) dựa trên bài báo MIRViT. Hệ thống cho phép:

- **Upload ảnh y tế** → tìm kiếm Top-6 ảnh tương tự nhất trong database
- **So sánh trực quan** giữa MIRViT (ViT-Small) và ResNet50 baseline
- **Giải thích mô hình** qua Attention Rollout (XAI)
- **Xem biểu đồ mAP** từ kết quả thực nghiệm trong bài báo

> ⚠️ **Lưu ý quan trọng:** Demo dùng **pretrained weights từ ImageNet** (không phải weights đã fine-tune trên dataset y tế thật). Ảnh trong database là **ảnh synthetic** được tạo bằng code. Mục đích là minh hoạ pipeline, không phải đánh giá hiệu năng thực tế.

---

## 2. Cấu trúc thư mục

```
ir/
├── demo/
│   ├── app.py                    # File chính — chạy demo Gradio
│   ├── generate_sample_data.py   # Tạo ảnh synthetic cho database
│   └── sample_images/            # Thư mục chứa ảnh mẫu (tự động tạo)
│       ├── covid/                # 15 ảnh X-quang COVID-19
│       ├── normal/               # 15 ảnh X-quang bình thường
│       ├── pneumonia/            # 15 ảnh X-quang viêm phổi
│       ├── melanoma/             # 12 ảnh da liễu melanoma
│       ├── nevi/                 # 12 ảnh nốt ruồi lành tính
│       ├── keratosis/            # 12 ảnh dày sừng
│       ├── polyp/                # 10 ảnh nội soi polyp
│       ├── normal_cecum/         # 10 ảnh manh tràng bình thường
│       ├── esophagitis/          # 10 ảnh viêm thực quản
│       └── ulcerative_colitis/   # 10 ảnh viêm loét đại tràng
├── MIRViT_summary.md             # Tóm tắt bài báo
├── MIRViT_code_report.md         # File này
└── .venv/                        # Virtual environment Python
```

**Tổng database:** 121 ảnh, 10 lớp bệnh, thuộc 3 nhóm dataset:

| Nhóm | Lớp bệnh | Số ảnh |
|------|----------|--------|
| 🫁 COVID-19 Chest X-Ray | covid, normal, pneumonia | 45 |
| 🔬 ISIC Skin Lesion | melanoma, nevi, keratosis | 36 |
| 🔭 Kvasir Endoscopy | polyp, normal_cecum, esophagitis, ulcerative_colitis | 40 |

---

## 3. Giải thích code chi tiết

### 3.1 Cấu hình & hằng số

```python
# demo/app.py — dòng 28–50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
IMG_SIZE = 224   # Kích thước ảnh đầu vào ViT (224×224)
TOP_K = 6        # Trả về 6 ảnh tương tự nhất
```

- **`DEVICE`**: Tự động dùng GPU nếu có, ngược lại dùng CPU.
- **`IMG_SIZE = 224`**: ViT-Small yêu cầu ảnh đầu vào 224×224 pixel.
- **`TOP_K = 6`**: Hiển thị 6 kết quả retrieval (2 hàng × 3 cột).

**Transform chuẩn hoá ảnh:**

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Mean ImageNet
        std=[0.229, 0.224, 0.225]     # Std ImageNet
    ),
])
```

Đây là chuẩn hoá ImageNet — phù hợp vì model dùng pretrained weights từ ImageNet.

---

### 3.2 Mô hình MIRViT

```python
# demo/app.py — dòng 65–87

class MIRViT(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        # Backbone: ViT-Small với patch 16×16
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,   # Tắt classification head, lấy embedding
        )
        # Projection head: 2 lớp Linear + GELU
        self.projector = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 384),
        )

    def forward(self, x):
        feat = self.backbone(x)        # CLS token embedding: (B, 384)
        proj = self.projector(feat)    # Projection: (B, 384)
        return F.normalize(proj, dim=-1)  # L2-normalize → cosine sim = dot product
```

**Giải thích từng phần:**

| Thành phần | Mô tả | Kích thước |
|-----------|-------|-----------|
| `vit_small_patch16_224` | ViT-Small, chia ảnh thành 14×14=196 patches, mỗi patch 16×16 px | ~22M params |
| `num_classes=0` | Bỏ classification head, lấy CLS token làm embedding | output: (B, 384) |
| `projector` | 2 lớp Linear + GELU — giúp học embedding tốt hơn cho contrastive loss | (384→384→384) |
| `F.normalize` | L2-normalize → cosine similarity = dot product (nhanh hơn) | output: unit vector |

**Tại sao dùng Projection Head?**  
Trong contrastive learning (SimCLR, MoCo...), projection head giúp backbone học được representation tốt hơn. Sau khi train, chỉ dùng backbone để lấy embedding, bỏ projection head. Trong demo này, cả hai được dùng cùng nhau vì không có fine-tuning.

---

### 3.3 Mô hình ResNet50 Baseline

```python
# demo/app.py — dòng 90–100

class ResNet50Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        # Bỏ lớp FC cuối (classification), giữ lại feature extractor
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        feat = self.features(x)    # Global Average Pooling output: (B, 2048, 1, 1)
        feat = feat.flatten(1)     # Flatten: (B, 2048)
        return F.normalize(feat, dim=-1)  # L2-normalize
```

**So sánh với MIRViT:**

| Tiêu chí | ResNet50 | MIRViT Small |
|----------|----------|--------------|
| Kiến trúc | CNN — local features | ViT — global attention |
| Embedding dim | 2048 | 384 |
| Receptive field | Cục bộ (local) | Toàn cục (global) |
| Giải thích | Khó (hộp đen) | Dễ (Attention Rollout) |
| mAP ISIC | 57.72% | **70.96%** |

---

### 3.4 Xây dựng Embedding Database

```python
# demo/app.py — dòng 120–166

def build_database():
    # Bước 1: Kiểm tra xem đã có ảnh mẫu chưa
    if not os.path.exists(SAMPLE_DIR) or ...:
        from generate_sample_data import generate_all_samples
        generate_all_samples()   # Tự động tạo ảnh nếu chưa có

    # Bước 2: Duyệt qua từng ảnh, trích xuất embedding
    for label in sorted(os.listdir(SAMPLE_DIR)):
        for fname in sorted(os.listdir(label_dir)):
            img = Image.open(fpath).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():   # Không tính gradient (inference mode)
                vit_emb = mirvit_model(tensor).cpu().numpy()[0]    # (384,)
                resnet_emb = resnet_model(tensor).cpu().numpy()[0] # (2048,)
            
            vit_embs.append(vit_emb)
            labels.append(label)
            paths.append(fpath)

    # Bước 3: Lưu vào numpy array để tính cosine similarity nhanh
    db_embeddings_vit = np.array(vit_embs)      # (121, 384)
    db_embeddings_resnet = np.array(resnet_embs) # (121, 2048)
```

**Pipeline xây dựng database:**

```
Ảnh mẫu (JPG)
    │
    ▼
Resize → 224×224
    │
    ▼
Normalize (ImageNet mean/std)
    │
    ├──► MIRViT → embedding (384-dim) ──► db_embeddings_vit  [121×384]
    │
    └──► ResNet50 → embedding (2048-dim) ──► db_embeddings_resnet [121×2048]
```

---

### 3.5 XAI — Attention Rollout

Đây là phần **giải thích mô hình** — cho thấy ViT "nhìn vào đâu" khi xử lý ảnh.

```python
# demo/app.py — dòng 172–213

def compute_attention_rollout(model, img_tensor):
    attentions_list = []
    
    # Bước 1: Gắn hook vào từng Attention block để thu thập attention weights
    for block in model.backbone.blocks:   # 12 blocks trong ViT-Small
        h = block.attn.register_forward_hook(
            lambda module, input, output: attentions_list.append(output.detach().cpu())
        )
    
    # Bước 2: Forward pass để thu thập attention
    with torch.no_grad():
        _ = model.backbone(img_tensor)
    
    # Bước 3: Thuật toán Attention Rollout
    result = torch.eye(seq_len)   # Ma trận đơn vị ban đầu
    for attn in attentions_list:  # Duyệt qua 12 layers
        attn_avg = attn.mean(dim=1)[0]              # Trung bình qua các heads
        attn_avg = attn_avg + torch.eye(seq_len)    # Cộng residual connection
        attn_avg = attn_avg / attn_avg.sum(dim=-1)  # Normalize
        result = torch.matmul(attn_avg, result)     # Tích luỹ qua các layers
    
    # Bước 4: Lấy attention từ CLS token → 196 patch tokens
    mask = result[0, 1:197]   # (196,) → reshape thành (14, 14)
    mask = mask.reshape(14, 14)
    return mask   # Heatmap 14×14 → resize lên 224×224 để overlay
```

**Thuật toán Attention Rollout** (Abnar & Zuidema, 2020):

```
Layer 1 attention: A₁ (197×197)
Layer 2 attention: A₂ (197×197)
...
Layer 12 attention: A₁₂ (197×197)

Rollout = A₁₂ × A₁₁ × ... × A₁   (tích ma trận)

→ Mỗi phần tử [i,j] = mức độ ảnh hưởng của token j lên token i
→ Lấy hàng CLS (index 0) → attention từ CLS đến 196 patches
→ Reshape (196,) → (14,14) → resize → (224,224)
```

**Tại sao dùng Attention Rollout thay vì chỉ xem attention layer cuối?**  
Attention layer cuối chỉ cho thấy attention ở bước cuối. Rollout tích luỹ qua tất cả 12 layers, phản ánh đầy đủ hơn những gì mô hình thực sự "chú ý" từ đầu đến cuối.

---

### 3.6 Retrieval Core

```python
# demo/app.py — dòng 330–410

def process_query(query_img, dataset_filter, show_xai):
    # Bước 1: Tiền xử lý ảnh query
    tensor = transform(query_pil).unsqueeze(0).to(DEVICE)
    
    # Bước 2: Trích xuất embedding của query
    with torch.no_grad():
        vit_emb = mirvit_model(tensor).cpu().numpy()[0]    # (384,)
        resnet_emb = resnet_model(tensor).cpu().numpy()[0] # (2048,)
    
    # Bước 3: Tính cosine similarity với toàn bộ database
    # Vì embedding đã L2-normalize → cosine_sim = dot product
    vit_sims = np.dot(filt_vit, vit_emb)       # (N,) — similarity scores
    resnet_sims = np.dot(filt_resnet, resnet_emb)
    
    # Bước 4: Xếp hạng và lấy Top-6
    vit_top = np.argsort(vit_sims)[::-1][:6]   # Indices của 6 ảnh tương tự nhất
    resnet_top = np.argsort(resnet_sims)[::-1][:6]
    
    # Bước 5: Hiển thị kết quả
    vit_fig = make_result_grid(vit_top, vit_sims, ...)
    resnet_fig = make_result_grid(resnet_top, resnet_sims, ...)
```

**Pipeline Retrieval:**

```
Query Image
    │
    ▼
Preprocess (resize, normalize)
    │
    ├──► MIRViT → query_emb (384-dim)
    │        │
    │        ▼
    │   Cosine Similarity với db_embeddings_vit (121×384)
    │        │
    │        ▼
    │   Argsort → Top-6 indices → Hiển thị ảnh
    │
    └──► ResNet50 → query_emb (2048-dim)
             │
             ▼
        Cosine Similarity với db_embeddings_resnet (121×2048)
             │
             ▼
        Argsort → Top-6 indices → Hiển thị ảnh
```

**Màu sắc similarity score:**
- 🟢 Xanh lá (`#00ff88`): sim > 0.8 — rất tương tự
- 🟡 Vàng (`#ffaa00`): 0.6 < sim ≤ 0.8 — tương tự vừa
- 🔴 Đỏ (`#ff6666`): sim ≤ 0.6 — ít tương tự

---

### 3.7 Giao diện Gradio

Demo có **4 tab** chính:

| Tab | Chức năng |
|-----|-----------|
| 🔍 Demo Retrieval | Upload ảnh → tìm kiếm → xem kết quả + XAI |
| 📊 So sánh Hiệu năng | Biểu đồ mAP từ bài báo |
| 🏗️ Kiến trúc & Phương pháp | Giải thích pipeline, loss function |
| 🧠 XAI — Giải thích mô hình | Phân tích Attention Rollout riêng |

**Luồng sự kiện Tab 1:**

```
User upload ảnh / click gallery
    │
    ▼
search_btn.click()
    │
    ▼
process_query(query_img, dataset_filter, show_xai)
    │
    ├──► vit_fig (matplotlib figure)
    ├──► resnet_fig (matplotlib figure)
    ├──► xai_fig (matplotlib figure, nếu show_xai=True)
    └──► stats_text (markdown text)
```

---

### 3.8 Tạo dữ liệu mẫu

File `generate_sample_data.py` tạo **ảnh synthetic** (giả lập) cho 3 nhóm bệnh:

#### Nhóm 1: X-quang phổi (`create_xray_image`)

```python
# Vẽ khung phổi + xương sườn trên nền đen
# COVID: thêm vùng mờ đục (ground glass opacity) — đặc trưng COVID-19
# Pneumonia: vùng đông đặc (consolidation) — đặc trưng viêm phổi
# Normal: mạch máu nhỏ — phổi sạch
# + Gaussian noise để giống ảnh thật hơn
```

#### Nhóm 2: Tổn thương da (`create_skin_lesion_image`)

```python
# Nền màu da (220, 180, 140)
# Melanoma: nhiều ellipse chồng chéo, màu tối, bờ không đều
# Nevi: ellipse tròn đều, màu đồng nhất — lành tính
# Keratosis: hình chữ nhật với texture thô ráp
```

#### Nhóm 3: Nội soi tiêu hoá (`create_endoscopy_image`)

```python
# Nền hồng của niêm mạc
# Polyp: khối tròn nhô lên
# Esophagitis: vùng đỏ bất thường rải rác
# Normal Cecum: nếp gấp đều đặn
# Ulcerative Colitis: nhiều vết loét nhỏ
```

**Số lượng ảnh được tạo:**

| Lớp | Số ảnh | Hàm tạo |
|-----|--------|---------|
| covid | 15 | `create_xray_image` |
| normal | 15 | `create_xray_image` |
| pneumonia | 15 | `create_xray_image` |
| melanoma | 12 | `create_skin_lesion_image` |
| nevi | 12 | `create_skin_lesion_image` |
| keratosis | 12 | `create_skin_lesion_image` |
| polyp | 10 | `create_endoscopy_image` |
| normal_cecum | 10 | `create_endoscopy_image` |
| esophagitis | 10 | `create_endoscopy_image` |
| ulcerative_colitis | 10 | `create_endoscopy_image` |
| **Tổng** | **121** | |

---

## 4. So sánh code với bài báo

Phần này đối chiếu **từng mục** trong `MIRViT_summary.md` với code thực tế trong `demo/app.py`.

---

### 4.1 Kiến trúc mô hình (Mục 2 bài báo)

| Nội dung bài báo | Trong code | Đánh giá |
|-----------------|-----------|----------|
| ViT-Small backbone, patch 16×16 | `timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)` | ✅ Đúng |
| Embedding dim 384 | `embed_dim=384`, `nn.Linear(384, 384)` | ✅ Đúng |
| Projection head 2 lớp Linear + activation | `nn.Linear → nn.GELU → nn.Linear` | ✅ Đúng |
| L2-normalize embedding | `F.normalize(proj, dim=-1)` | ✅ Đúng |
| Cosine similarity cho retrieval | `np.dot(filt_vit, vit_emb)` — dot product của unit vectors = cosine sim | ✅ Đúng |
| Fine-tune ViT với contrastive loss | Dùng ImageNet pretrained, **không có training loop** | ❌ Miss (hợp lý cho demo) |
| **Contrastive Loss** (`L_contr`) | **Không implement** — không có loss function nào | ❌ Miss |
| **KoLeo Regularization** (`L_KoLeo`) | **Không implement** — chỉ đề cập trong UI text | ❌ Miss |
| **Cross-batch memory** cho contrastive | **Không có** | ❌ Miss |
| **λ = 0.3** (hyperparameter tối ưu) | Chỉ đề cập trong markdown text Tab 3, không dùng trong code | ⚠️ Chỉ mention, không dùng |
| MIRdeiT Small (DeiT-Small backbone) | **Không có** — chỉ có MIRViT Small | ❌ Miss |
| MIRViT Base (ViT-Base, 786-dim) | **Không có** — chỉ có MIRViT Small | ❌ Miss |

> **Lưu ý về embedding dim MIRViT Base:** Bài báo ghi **786-dim** nhưng ViT-Base thực tế có embedding **768-dim** (không phải 786). Đây có thể là lỗi đánh máy trong bài báo. Code demo dùng đúng 384-dim cho ViT-Small.

---

### 4.2 Bộ dữ liệu (Mục 3 bài báo)

| Nội dung bài báo | Trong code | Đánh giá |
|-----------------|-----------|----------|
| COVID-19: COVID, bình thường, viêm phổi | `["covid", "normal", "pneumonia"]` | ✅ Đúng 3 lớp |
| ISIC 2017: nevi, keratosis, melanoma | `["melanoma", "nevi", "keratosis"]` | ✅ Đúng 3 lớp |
| Kvasir-V2: **8 lớp** nội soi | `["polyp", "normal_cecum", "esophagitis", "ulcerative_colitis"]` — **chỉ 4 lớp** | ⚠️ Thiếu 4 lớp |
| COVID: ~9,208 ảnh | 45 ảnh synthetic | ⚠️ Demo dùng ảnh giả lập |
| ISIC: 2,750 ảnh | 36 ảnh synthetic | ⚠️ Demo dùng ảnh giả lập |
| Kvasir: 8,000 ảnh | 40 ảnh synthetic | ⚠️ Demo dùng ảnh giả lập |

**4 lớp Kvasir-V2 bị thiếu trong demo:**
Bài báo dùng Kvasir-V2 với 8 lớp. Demo chỉ implement 4 lớp:

| Có trong demo | Thiếu trong demo |
|--------------|-----------------|
| polyp | dyed-lifted-polyps |
| normal_cecum | dyed-resection-margins |
| esophagitis | pylorus |
| ulcerative_colitis | z-line |

---

### 4.3 Kết quả mAP (Mục 4 bài báo)

| Dataset | ResNet50 | DenseNet121 | DeiT-Small (CE) | MIRViT Small | Đánh giá |
|---------|:--------:|:-----------:|:---------------:|:------------:|----------|
| ISIC | 57.72% | 58.38% | 63.32% | **70.96%** | ✅ Khớp chính xác |
| COVID | 91.33% | 94.62% | 92.93% | **96.96%** | ✅ Khớp chính xác |
| Kvasir | 84.85% | 83.89% | 88.15% | **90.16%** | ✅ Khớp chính xác |

Các con số mAP trong `create_map_comparison_chart()` và bảng markdown Tab 2 **khớp 100%** với bài báo.

**Tuy nhiên, bài báo còn có kết quả của các model khác không được đưa vào biểu đồ:**
- MIRViT Base (786-dim) — không có trong biểu đồ demo
- MIRdeiT Small — không có trong biểu đồ demo  
- MedViT (hybrid CNN-Transformer) — không có trong biểu đồ demo

---

### 4.4 XAI — Khả năng giải thích (Mục 5 bài báo)

| Nội dung bài báo | Trong code | Đánh giá |
|-----------------|-----------|----------|
| **Attention Rollout** | `compute_attention_rollout()` — implement đầy đủ | ✅ Có |
| **Chefer2** (gradient-weighted) | **Không implement** | ❌ Miss |
| **TIS** (Transformer Input Sampling) | **Không implement** | ❌ Miss — quan trọng nhất theo bài báo |
| **ViT-CX** (counterfactual) | **Không implement** | ❌ Miss |
| **TAMs, BTH, BTT** | **Không implement** | ❌ Miss |
| Chỉ số Insertion AUC | **Không tính** — chỉ hiển thị heatmap | ❌ Miss |
| Chỉ số Deletion AUC | **Không tính** — chỉ hiển thị heatmap | ❌ Miss |
| Bảng so sánh XAI (AUC scores) | Hiển thị trong Tab 4 dưới dạng markdown | ✅ Có (chỉ text, không tính thực) |

**Điểm quan trọng:** Bài báo kết luận **TIS là phương pháp XAI tốt nhất** về mặt định tính, nhưng demo lại chỉ implement **Attention Rollout** — phương pháp đơn giản nhất. Đây là điểm miss đáng chú ý nhất về XAI.

---

### 4.5 Baseline models (Mục 4 bài báo)

| Baseline trong bài báo | Trong code | Đánh giá |
|-----------------------|-----------|----------|
| ResNet50 | `ResNet50Baseline` class | ✅ Có |
| DenseNet121 | **Không implement** | ❌ Miss |
| DeiT-Small (Cross-entropy) | **Không implement** | ❌ Miss |
| MedViT | **Không implement** | ❌ Miss |

---

### 4.6 Tổng hợp: Những điểm MISS quan trọng

| # | Điểm miss | Mức độ quan trọng | Lý do hợp lý? |
|---|-----------|:-----------------:|---------------|
| 1 | Không có training loop (Contrastive Loss + KoLeo) | 🔴 Cao | ✅ Demo chỉ inference |
| 2 | Thiếu TIS — XAI tốt nhất theo bài báo | 🔴 Cao | ⚠️ Nên thêm vào demo |
| 3 | Kvasir-V2 chỉ 4/8 lớp | 🟡 Trung bình | ⚠️ Có thể bổ sung |
| 4 | Thiếu Chefer2, ViT-CX cho so sánh XAI | 🟡 Trung bình | ✅ Phức tạp, hợp lý bỏ qua |
| 5 | Không có MIRdeiT Small, MIRViT Base | 🟡 Trung bình | ✅ Đủ để demo với MIRViT Small |
| 6 | Không tính Insertion/Deletion AUC | 🟡 Trung bình | ✅ Cần dataset thật |
| 7 | Thiếu DenseNet121, DeiT-Small baseline | 🟢 Thấp | ✅ Biểu đồ đã có số liệu |
| 8 | Cross-batch memory cho contrastive | 🟢 Thấp | ✅ Không train |
| 9 | Dùng ImageNet weights thay vì fine-tuned | 🔴 Cao | ✅ Demo không cần train |

---

### 📝 Nhận xét tổng thể

**Điểm mạnh của demo:**
- Kiến trúc MIRViT Small (backbone + projection head + L2-normalize) được implement **chính xác**
- Pipeline retrieval (embedding → cosine similarity → ranking) **đúng hoàn toàn**
- Số liệu mAP trong biểu đồ **khớp 100%** với bài báo
- Attention Rollout được implement **đúng thuật toán**
- Giao diện trực quan, dễ hiểu, phù hợp để demo

**Điểm cần lưu ý khi trình bày:**
- Demo **không có training** → kết quả retrieval dựa trên ImageNet features, không phải medical features đã fine-tune
- Demo chỉ có **Attention Rollout**, trong khi bài báo kết luận **TIS** là XAI tốt nhất
- Kvasir-V2 trong demo chỉ có **4/8 lớp** so với bài báo
- Ảnh trong database là **synthetic** (tạo bằng code), không phải ảnh y tế thật

---

## 5. Hướng dẫn chạy trên local

### Yêu cầu hệ thống

- **Python:** 3.9 (khuyến nghị) hoặc 3.10
- **RAM:** Tối thiểu 4GB (8GB khuyến nghị)
- **Disk:** ~500MB (cho model weights)
- **GPU:** Không bắt buộc (CPU cũng chạy được, chậm hơn ~3-5x)
- **OS:** macOS, Linux, Windows

---

### Bước 1: Clone / Mở thư mục project

```bash
cd /Users/hado/Downloads/ai_fpt/ir
```

---

### Bước 2: Tạo môi trường ảo Python

```bash
python3 -m venv .venv
```

Kích hoạt môi trường:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Sau khi kích hoạt, terminal sẽ hiện `(.venv)` ở đầu dòng.

---

### Bước 3: Cài đặt dependencies

> ⚠️ **Quan trọng:** Phải cài đúng version để tránh lỗi tương thích.

```bash
pip install --upgrade pip

pip install \
  numpy pillow matplotlib timm torch torchvision \
  gradio==4.44.1 \
  huggingface_hub==0.25.2 \
  pydantic==2.10.6 \
  fastapi==0.115.6 \
  starlette==0.41.3 \
  "urllib3<2"
```

**Giải thích tại sao phải pin version:**

| Package | Version | Lý do |
|---------|---------|-------|
| `gradio` | 4.44.1 | Version mới hơn có lỗi API schema với Python 3.9 |
| `huggingface_hub` | 0.25.2 | Version mới hơn bỏ `HfFolder` mà gradio 4.44.1 cần |
| `pydantic` | 2.10.6 | Version mới hơn không tương thích với fastapi 0.115.6 |
| `fastapi` | 0.115.6 | Tương thích với gradio 4.44.1 |
| `starlette` | 0.41.3 | Tương thích với fastapi 0.115.6 |
| `urllib3` | <2 | Tránh cảnh báo LibreSSL trên macOS |

---

### Bước 4: Chạy ứng dụng

```bash
python demo/app.py
```

**Lần đầu chạy** sẽ tự động:
1. Download model ViT-Small (~88MB từ HuggingFace)
2. Download model ResNet50 (~98MB từ PyTorch)
3. Tạo 121 ảnh synthetic trong `demo/sample_images/`
4. Trích xuất embeddings cho toàn bộ database

**Output mong đợi:**

```
⏳ Đang load models...
model.safetensors: 100%|████| 88.2M/88.2M [00:08<00:00]
✅ Models loaded! Device: cpu
⏳ Đang xây dựng embedding database...
✅ Database: 121 ảnh, 10 lớp

============================================================
🏥 MIRViT Demo — Medical Image Retrieval
============================================================
📊 Database: 121 ảnh | 10 lớp
🖥️  Device: cpu
🌐 URL: http://localhost:7860
============================================================

Running on local URL:  http://127.0.0.1:7860
```

---

### Bước 5: Mở trình duyệt

Truy cập:

```
http://127.0.0.1:7860
```

hoặc:

```
http://localhost:7860
```

---

### Bước 6: Sử dụng demo

#### Tab 🔍 Demo Retrieval

1. **Upload ảnh** bằng cách kéo thả vào ô "Ảnh Query"  
   **hoặc** click vào một trong các ảnh mẫu trong gallery bên dưới
2. Chọn **dataset** muốn tìm kiếm (hoặc để "Tất cả")
3. Tick/bỏ tick **"Hiển thị XAI"** tuỳ ý
4. Nhấn **🔍 Tìm kiếm ảnh tương tự**
5. Xem kết quả:
   - **Cột trái:** Top-6 kết quả của MIRViT Small
   - **Cột phải:** Top-6 kết quả của ResNet50 Baseline
   - **Bên dưới:** Attention Rollout heatmap (nếu bật XAI)

#### Tab 📊 So sánh Hiệu năng

- Nhấn **"Hiển thị biểu đồ so sánh mAP"** để xem biểu đồ bar chart
- Xem bảng kết quả chi tiết từ bài báo

#### Tab 🧠 XAI

- Upload ảnh bất kỳ
- Nhấn **"Phân tích Attention Rollout"**
- Xem 3 panel: ảnh gốc | attention heatmap | overlay

---

### Dừng ứng dụng

Nhấn `Ctrl + C` trong terminal.

---

## 6. Troubleshooting

### Lỗi: `ImportError: cannot import name 'HfFolder'`

```bash
pip install "huggingface_hub==0.25.2"
```

### Lỗi: `TypeError: argument of type 'bool' is not iterable`

Lỗi trong `gradio_client/utils.py` — do version pydantic/fastapi quá mới:

```bash
pip install "pydantic==2.10.6" "fastapi==0.115.6" "starlette==0.41.3"
```

### Lỗi: `ValueError: When localhost is not accessible, a shareable link must be created`

Kiểm tra `demo/app.py` cuối file, đảm bảo:

```python
demo.launch(
    server_name="127.0.0.1",   # Không phải "0.0.0.0"
    server_port=7860,
    share=False,
    show_error=True,
    show_api=False,
    inbrowser=True,
)
```

### Lỗi: Port 7860 đã bị chiếm

```bash
# Kiểm tra process đang dùng port 7860
lsof -i :7860

# Hoặc đổi port trong app.py
server_port=7861
```

### Chạy chậm trên CPU

Bình thường — mỗi lần tìm kiếm mất ~2-5 giây trên CPU. Nếu có GPU NVIDIA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Cảnh báo `NotOpenSSLWarning` (macOS)

Không ảnh hưởng đến chức năng. Đã được xử lý bằng `urllib3<2`.

---

## Tóm tắt nhanh (Quick Start)

```bash
# 1. Vào thư mục project
cd /Users/hado/Downloads/ai_fpt/ir

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate

# 3. Chạy app
python demo/app.py

# 4. Mở trình duyệt tại http://127.0.0.1:7860
```

---

*Báo cáo được tạo tự động dựa trên phân tích source code `demo/app.py` và `demo/generate_sample_data.py`.*
