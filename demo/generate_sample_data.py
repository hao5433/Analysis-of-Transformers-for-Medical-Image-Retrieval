"""
Script tạo dữ liệu mẫu giả lập cho demo MIRViT
Tạo ảnh y tế synthetic để demo không cần download dataset thật
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

def create_xray_image(label, size=(224, 224)):
    """Tạo ảnh X-quang phổi giả lập"""
    img = Image.new("RGB", size, color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    # Vẽ khung phổi
    cx, cy = size[0] // 2, size[1] // 2

    # Phổi trái
    draw.ellipse([cx - 90, cy - 70, cx - 10, cy + 80], outline=(180, 180, 180), width=2)
    # Phổi phải
    draw.ellipse([cx + 10, cy - 70, cx + 90, cy + 80], outline=(180, 180, 180), width=2)
    # Xương sườn
    for i in range(5):
        y = cy - 50 + i * 25
        draw.arc([cx - 95, y, cx + 95, y + 30], start=0, end=180, fill=(120, 120, 120), width=1)

    if label == "covid":
        # Thêm vùng mờ đục (ground glass opacity) - đặc trưng COVID
        for _ in range(8):
            x = random.randint(cx - 80, cx + 80)
            y = random.randint(cy - 60, cy + 70)
            r = random.randint(10, 25)
            alpha = random.randint(60, 120)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(alpha, alpha, alpha))
        # Thêm text label
        draw.text((5, 5), "COVID-19", fill=(255, 100, 100))

    elif label == "pneumonia":
        # Vùng đông đặc (consolidation) - đặc trưng viêm phổi
        draw.ellipse([cx - 70, cy, cx - 20, cy + 60], fill=(100, 100, 100))
        draw.ellipse([cx + 20, cy - 20, cx + 75, cy + 50], fill=(90, 90, 90))
        draw.text((5, 5), "Pneumonia", fill=(255, 200, 100))

    elif label == "normal":
        # Phổi sạch, thêm chi tiết mạch máu
        for _ in range(6):
            x1 = random.randint(cx - 80, cx + 80)
            y1 = random.randint(cy - 60, cy + 70)
            x2 = x1 + random.randint(-20, 20)
            y2 = y1 + random.randint(10, 30)
            draw.line([x1, y1, x2, y2], fill=(80, 80, 80), width=1)
        draw.text((5, 5), "Normal", fill=(100, 255, 100))

    # Thêm noise để giống ảnh thật
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def create_skin_lesion_image(label, size=(224, 224)):
    """Tạo ảnh tổn thương da giả lập"""
    # Nền da
    skin_color = (220, 180, 140)
    img = Image.new("RGB", size, color=skin_color)
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2

    if label == "melanoma":
        # Tổn thương không đều, màu tối, bờ không đều
        for i in range(5):
            offset_x = random.randint(-15, 15)
            offset_y = random.randint(-15, 15)
            r = random.randint(30, 55)
            color = (random.randint(30, 80), random.randint(20, 50), random.randint(20, 40))
            draw.ellipse([cx + offset_x - r, cy + offset_y - r,
                         cx + offset_x + r, cy + offset_y + r], fill=color)
        draw.text((5, 5), "Melanoma", fill=(255, 50, 50))

    elif label == "nevi":
        # Nốt ruồi lành tính - tròn đều, màu đồng nhất
        color = (100, 60, 40)
        r = random.randint(25, 40)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        draw.text((5, 5), "Nevi", fill=(50, 50, 255))

    elif label == "keratosis":
        # Dày sừng - bề mặt thô ráp
        color = (160, 100, 60)
        draw.rectangle([cx - 40, cy - 30, cx + 40, cy + 30], fill=color)
        for _ in range(20):
            x = random.randint(cx - 40, cx + 40)
            y = random.randint(cy - 30, cy + 30)
            draw.point([x, y], fill=(120, 70, 40))
        draw.text((5, 5), "Keratosis", fill=(255, 165, 0))

    # Thêm texture da
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def create_endoscopy_image(label, size=(224, 224)):
    """Tạo ảnh nội soi tiêu hoá giả lập"""
    # Nền hồng của niêm mạc
    base_colors = {
        "polyp": (180, 80, 80),
        "normal_cecum": (200, 120, 100),
        "esophagitis": (160, 60, 60),
        "ulcerative_colitis": (140, 50, 50),
    }
    base = base_colors.get(label, (190, 100, 90))
    img = Image.new("RGB", size, color=base)
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2

    if label == "polyp":
        # Polyp - khối nhô lên
        draw.ellipse([cx - 35, cy - 35, cx + 35, cy + 35],
                    fill=(220, 100, 100), outline=(255, 150, 150), width=2)
        draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(240, 120, 120))
        draw.text((5, 5), "Polyp", fill=(255, 255, 100))

    elif label == "esophagitis":
        # Viêm thực quản - vùng đỏ bất thường
        for _ in range(6):
            x = random.randint(20, size[0] - 20)
            y = random.randint(20, size[1] - 20)
            draw.ellipse([x-15, y-8, x+15, y+8], fill=(220, 50, 50))
        draw.text((5, 5), "Esophagitis", fill=(255, 255, 100))

    elif label == "normal_cecum":
        # Manh tràng bình thường - nếp gấp đều
        for i in range(4):
            y = 40 + i * 45
            draw.arc([20, y, size[0]-20, y+30], start=0, end=180,
                    fill=(170, 90, 80), width=2)
        draw.text((5, 5), "Normal Cecum", fill=(100, 255, 100))

    elif label == "ulcerative_colitis":
        # Viêm loét đại tràng
        for _ in range(10):
            x = random.randint(10, size[0] - 10)
            y = random.randint(10, size[1] - 10)
            r = random.randint(5, 15)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(100, 30, 30))
        draw.text((5, 5), "Ulcerative Colitis", fill=(255, 255, 100))

    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 6, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def generate_all_samples():
    """Tạo tất cả ảnh mẫu"""
    print("🔄 Đang tạo dữ liệu mẫu...")

    # COVID X-Ray dataset
    configs = [
        ("covid", 15, create_xray_image),
        ("normal", 15, create_xray_image),
        ("pneumonia", 15, create_xray_image),
        ("melanoma", 12, create_skin_lesion_image),
        ("nevi", 12, create_skin_lesion_image),
        ("keratosis", 12, create_skin_lesion_image),
        ("polyp", 10, create_endoscopy_image),
        ("normal_cecum", 10, create_endoscopy_image),
        ("esophagitis", 10, create_endoscopy_image),
        ("ulcerative_colitis", 10, create_endoscopy_image),
    ]

    for label, count, fn in configs:
        folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(count):
            img = fn(label)
            img.save(os.path.join(folder, f"{label}_{i:03d}.jpg"))
        print(f"  ✅ {label}: {count} ảnh")

    print(f"\n✅ Hoàn thành! Ảnh được lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_samples()
