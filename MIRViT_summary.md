# Tóm Tắt: Analysis of Transformers for Medical Image Retrieval (MIRViT)

> **Tác giả:** Arvapalli Sai Susmitha (IIT Kanpur) & Vinay P. Namboodiri (University of Bath)
> **Hội nghị:** MIDL 2024 (Under Review)

---

## 1. Vấn đề nghiên cứu

Truy xuất ảnh y tế dựa trên nội dung (Content-based Medical Image Retrieval – MIR) giúp bác sĩ tìm kiếm các ca bệnh tương tự trong cơ sở dữ liệu để hỗ trợ chẩn đoán. Các phương pháp truyền thống dựa trên **CNN** gặp hai hạn chế lớn:
- Khó nắm bắt các phụ thuộc tầm xa trong ảnh
- Thiếu khả năng giải thích ("hộp đen")

Bài báo đề xuất sử dụng **Vision Transformers (ViT)** kết hợp **Contrastive Learning** để khắc phục các vấn đề trên.

---

## 2. Phương pháp (MIRViT)

### 2.1 Kiến trúc tổng thể

| Giai đoạn | Mô tả |
|-----------|-------|
| **Huấn luyện** | Fine-tune ViT với contrastive loss + differential entropy regularization |
| **Kiểm tra** | Dùng mô hình đã huấn luyện để trích xuất embedding, xếp hạng ảnh qua cosine similarity |

### 2.2 Hàm mất mát

**Tổng loss:** `L = L_contr + λ * L_KoLeo`

- **Contrastive Loss** (`L_contr`): Kéo ảnh cùng lớp lại gần nhau, đẩy ảnh khác lớp ra xa. Sử dụng *cross-batch memory* để tăng số lượng negative pair mà không tốn quá nhiều bộ nhớ.
- **KoLeo Regularization** (`L_KoLeo`): Tối đa hoá khoảng cách giữa mỗi điểm embedding và láng giềng gần nhất, tránh các biểu diễn bị chồng chéo.

### 2.3 Các biến thể mô hình

| Mô hình | Kiến trúc gốc | Embedding | Patch size |
|---------|--------------|-----------|------------|
| MIRViT small | ViT-Small | 384 | 16×16 |
| MIRdeiT small | DeiT-Small | 384 | 16×16 |
| MIRViT base | ViT-Base | 786 | 16×16 |

---

## 3. Bộ dữ liệu thực nghiệm

| Dataset | Mô tả | Số ảnh |
|---------|-------|--------|
| **COVID-19 Chest X-Ray** | X-quang phổi: COVID, bình thường, viêm phổi | ~9,208 |
| **ISIC 2017** | Tổn thương da: nevi, keratosis, melanoma | 2,750 |
| **Kvasir-V2** | Nội soi tiêu hoá, 8 lớp | 8,000 |

---

## 4. Kết quả

### 4.1 So sánh kiến trúc (mAP)

**MIRViT small vượt trội trên cả 3 bộ dữ liệu:**

| Dataset | Resnet50 | DenseNet121 | deit small (CE) | **MIRViT small** |
|---------|----------|-------------|-----------------|-----------------|
| ISIC | 57.72% | 58.38% | 63.32% | **70.96%** |
| COVID | 91.33% | 94.62% | 92.93% | **96.96%** |
| Kvasir | 84.85% | 83.89% | 88.15% | **90.16%** |

> Ví dụ trên ISIC: MIRViT small tăng **13.24% mAP** so với ResNet50 và **7.64%** so với deit small dùng cross-entropy loss.

### 4.2 Nhận xét về siêu tham số

- **MIRViT small** (embedding 384) liên tục vượt MIRViT base (embedding 786) → embedding nhỏ hơn phù hợp hơn cho retrieval y tế
- **λ = 0.3** là lựa chọn tối ưu cho hầu hết các bộ dữ liệu
- **MedViT** (hybrid CNN-Transformer) mạnh về phân loại nhưng kém trong retrieval

---

## 5. Đánh giá khả năng giải thích (XAI)

### Các phương pháp được so sánh
- Attention Rollout, Chefer2, TAMs, BTH, BTT, ViT-CX, **TIS (Transformer Input Sampling)**

### Chỉ số đánh giá (AUC)
- **Insertion:** AUC càng cao càng tốt (ảnh dần hiện ra từ vùng quan trọng → similarity tăng nhanh)
- **Deletion:** AUC càng thấp càng tốt (ảnh dần mất đi từ vùng quan trọng → similarity giảm nhanh)

### Kết quả (MIRViT small)

| Dataset | Metric | Chefer2 | Rollout | **TIS** | ViT-CX |
|---------|--------|---------|---------|---------|--------|
| ISIC | Insertion | **0.79** | 0.78 | 0.78 | 0.72 |
| ISIC | Deletion | **0.41** | 0.45 | **0.41** | 0.53 |
| COVID | Insertion | **0.70** | 0.69 | 0.67 | 0.62 |
| Kvasir | Deletion | **0.40** | 0.42 | **0.42** | 0.49 |

> **TIS** nổi bật nhất về mặt **định tính** – saliency map tập trung rõ vào vùng tổn thương, trong khi Chefer2 mạnh hơn một chút về số liệu định lượng.

---

## 6. Kết luận

| Điểm chính | Chi tiết |
|------------|---------|
| Mô hình tốt nhất | **MIRViT small** với λ = 0.3 |
| Loss tốt nhất | **Contrastive loss** (có hoặc không có KoLeo regularization) vượt cross-entropy |
| XAI tốt nhất | **TIS** – trực quan và dễ diễn giải nhất trong môi trường lâm sàng |
| Hướng tương lai | Tiếp tục khám phá các cải tiến thuật toán cho bài toán MIR y tế |

---

*Bài báo cung cấp bằng chứng thực nghiệm toàn diện rằng Vision Transformers với contrastive learning là hướng đi tiềm năng cho truy xuất ảnh y tế, đồng thời giải quyết vấn đề "hộp đen" thông qua các kỹ thuật XAI.*
