# 🚀 Hướng dẫn Triển khai Huấn luyện (LVTN 2026)

Tài liệu này cung cấp các bước thiết lập môi trường và khởi chạy quy trình huấn luyện cho dự án **Multimodal Clustering-Guided Negative Sampling** trên hai nền tảng Kaggle và Google Colab.

---

## 🏗️ 1. Triển khai trên Kaggle
Sử dụng khi bạn muốn huấn luyện trực tiếp từ mã nguồn mới nhất trên GitHub.

```bash
# 1. Clone repository
!git clone https://github.com/lznguyn/LVTN_2026.git
%cd LVTN_2026

# 2. Cài đặt thư viện (Chế độ im lặng)
!pip install -q -r requirements.txt

# 3. Tiền xử lý dữ liệu & Phân cụm
!python scripts/prepare_dataset.py
!export HF_ENDPOINT=https://hf-mirror.com && python scripts/create_clusters.py

# 4. Bắt đầu huấn luyện
print("--- ĐANG BẮT ĐẦU TRAIN ---")
!export PYTHONPATH=$PYTHONPATH:$(pwd) HF_ENDPOINT=https://hf-mirror.com && python scripts/train.py
```

---

## 🏗️ 2. Triển khai trên Google Colab
Sử dụng khi bạn đã tải mã nguồn về Drive dưới dạng file ZIP và muốn huấn luyện trên Colab.

```bash
# 1. Kết nối với Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Giải nén mã nguồn vào SSD của Colab (Tăng tốc độ đọc/ghi)
!rm -rf /content/Multimodal
!unzip -q "/content/drive/MyDrive/Multimodal_Negative_Sampling.zip" -d "/content/Multimodal"
%cd /content/Multimodal/LVTN_2026

# 3. Cài đặt các thư viện bổ trợ
!pip install -q -r requirements.txt
!pip install -q nltk timm transformers
import nltk
nltk.download('punkt')

# 4. Chuẩn bị dữ liệu và Phân cụm
print("--- ĐANG TỰ ĐỘNG TẢI DỮ LIỆU TỪ WEB (IU-XRAY) ---")
!python scripts/prepare_dataset.py
print("--- ĐANG TẠO CÁC CỤM BỆNH LÝ (CLUSTERS) ---")
!python scripts/create_clusters.py

# 5. Chạy script huấn luyện chính
print("--- ĐANG BẮT ĐẦU TRAIN ---")
!export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/train.py
```

---
**💡 Chú ý:** Trong quá trình huấn luyện, nếu bạn muốn thay đổi các thông số như `Learning Rate` hay `Batch Size`, hãy chỉnh sửa trực tiếp trong file `configs/default.yaml`.
