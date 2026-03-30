# 🚀 HƯỚNG DẪN CHẠY LUẬN VĂN MULTIMODAL - BẢN TỐI ƯU CUỐI CÙNG 🚀
*Hỗ trợ: GPU T4 Colab | Dữ liệu: MIMIC-CXR | Kỹ thuật: Mixed Precision + Gradient Accumulation*

---

## 🛠️ BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG DRIVE
1. Trên Google Drive: Chuột phải vào thư mục `db_MIMIC_CXRCXR` (trong mục Shared with me) -> **Tổ chức (Organize)** -> **Thêm lối tắt (Add Shortcut)** vào **Drive của tôi (My Drive)**.
2. Nén Code thành file `Multimodal_Negative_Sampling.zip` và tải lên ổ gốc Drive.

---

## 💻 BƯỚC 2: CÁC BLOCK CODE TRÊN COLAB (Copy & Paste)

### 🔵 Cell 1: Mount Drive và Giải nén
```python
from google.colab import drive
drive.mount('/content/drive')

# Giải nén và chuyển vào thư mục dự án
!rm -rf /content/Multimodal
!unzip -q "/content/drive/MyDrive/Multimodal_Negative_Sampling.zip" -d "/content/Multimodal"

# Di chuyển nếu bị lồng thư mục (Fix lỗi Pathing)
import os
if os.path.exists('/content/Multimodal/Multimodal_Negative_Sampling'):
    !mv /content/Multimodal/Multimodal_Negative_Sampling/* /content/Multimodal/
    !rm -rf /content/Multimodal/Multimodal_Negative_Sampling
```

### 🔵 Cell 2: Cài đặt Thư viện
```bash
%cd /content/Multimodal
!pip install -r requirements.txt
```

### 🔵 Cell 3: Cấu hình YAML Tối ƯU (Để tránh nổ RAM GPU)
*Hãy chạy Cell này để nó tự động tạo file cấu hình chuẩn nhất cho bạn*
```python
import yaml

config = {
    'data': {
        'train_csv': 'data/splits/train.csv',
        'val_csv': 'data/splits/val.csv',
        'image_size': 256,
        'max_text_length': 128
    },
    'model': {
        'embed_dim': 512,
        'image_encoder': 'swinv2_base_window12to16_192to256',
        'text_encoder': 'emilyalsentzer/Bio_ClinicalBERT',
        'temperature': 0.07
    },
    'training': {
        'batch_size': 8,              # Chạy nhỏ để ko OOM
        'gradient_accumulation_steps': 4, # Tổng batch vẫn là 32 (8x4)
        'epochs': 50,                 # Mục tiêu 50 Epoch
        'lr': 0.0001,
        'weight_decay': 0.01,
        'num_workers': 2,
        'max_steps_per_epoch': 800,   # Chốt 1 Epoch nhanh sau 800 bước
        'log_every_n_steps': 10,
        'checkpoint_dir': 'checkpoints/'
    }
}

with open('configs/default.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f)
print("✅ Đã tạo file Cấu hình Tối ưu!")
```

### 🔵 Cell 4: Tiền xử lý dữ liệu MIMIC-CXR
```bash
# Nhớ dán đúng mã nguồn prepare_mimic_colab.py đã sửa vào file trước khi chạy
!python scripts/prepare_mimic_colab.py
!python scripts/create_clusters.py
```

### 🔵 Cell 5: HUẤN LUYỆN (Siêu tốc độ)
```bash
# Thiết lập đường dẫn hệ thống để tránh lỗi "No module named src"
%cd /content/Multimodal
!export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/train.py
```

---

## 🌟 CÁC LƯU Ý SỐNG CÒN:

1. **Lỗi CUDA Out of Memory:** Nếu bị, hãy vào Menu `Runtime` -> `Restart Session` (Khởi động lại phiên) rồi chạy lại Block 5.
2. **Lỗi Half Precision (Overflow):** Đảm bảo trong file `src/losses/contrastive.py` bạn đã sửa con số `-1e9` thành `-10000.0`.
3. **Theo dõi:** Nếu thấy thanh tiến trình hiện `800/800` là xong 1 Epoch và hiện `⭐ [CÓ CẢI THIỆN]`, chúc mừng bạn đã thành công nạp được "Tri thức" vào model!

*(Bạn có thể Copy nội dung này lưu vào file `HD_THU_NGHIEM.md` trên máy mình nhé!)*
