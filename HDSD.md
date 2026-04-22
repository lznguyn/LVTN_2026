# 🚀 Hướng dẫn Triển khai Huấn luyện (LVTN 2026)

Tài liệu này cung cấp các bước thiết lập môi trường và khởi chạy quy trình huấn luyện cho dự án **Multimodal Clustering-Guided Negative Sampling** trên hai nền tảng Kaggle và Google Colab.

---

## 🏗️ 1. Triển khai trên Kaggle
Sử dụng khi bạn muốn huấn luyện trực tiếp từ mã nguồn mới nhất trên GitHub.

```bash
# 1. Clone repository
!git clone [https://github.com/lznguyn/LVTN_2026.git](https://github.com/lznguyn/LVTN_2026.git)
%cd LVTN_2026

# 2. Cài đặt thư viện (Chế độ im lặng)
!pip install -q -r requirements.txt

# 3. Tiền xử lý dữ liệu & Phân cụm
# Sử dụng mirror nếu gặp vấn đề kết nối HuggingFace
!python scripts/prepare_dataset.py
!export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com) && python scripts/create_clusters.py

# 4. Bắt đầu huấn luyện
print("--- ĐANG BẮT ĐẦU TRAIN ---")
!export PYTHONPATH=$PYTHONPATH:$(pwd) HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com) && python scripts/train.py

## 🏗️ 2. Triển khai trên gg colab

from google.colab import drive
drive.mount('/content/drive')

# Xóa thư mục cũ nếu có và giải nén vào SSD của Colab
!rm -rf /content/Multimodal
!unzip -q "/content/drive/MyDrive/Multimodal_Negative_Sampling.zip" -d "/content/Multimodal"
%cd /content/Multimodal/LVTN_2026

# Cài đặt các thư viện bổ trợ
!pip install -q -r requirements.txt
!pip install -q nltk timm transformers

import nltk
nltk.download('punkt')

# Tải tập dữ liệu IU-XRAY
print("--- ĐANG TỰ ĐỘNG TẢI DỮ LIỆU TỪ WEB (IU-XRAY) ---")
!python scripts/prepare_dataset.py

# Tạo các cụm bệnh lý (Clusters)
print("--- ĐANG TẠO CÁC CỤM BỆNH LÝ (CLUSTERS) ---")
!python scripts/create_clusters.py
print("\n✅ XONG! Dữ liệu đã sẵn sàng trên SSD của Colab.")

# Chạy script huấn luyện chính
print("--- ĐANG BẮT ĐẦU TRAIN ---")
!export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/train.py
