//kaggle
!git clone https://github.com/lznguyn/LVTN_2026.git
%cd LVTN_2026
!pip install -r requirements.txt
!python scripts/prepare_dataset.py
!export HF_ENDPOINT=https://hf-mirror.com && python scripts/create_clusters.py
# 6. BẮT ĐẦU HUẤN LUYỆN
print("--- ĐANG BẮT ĐẦU TRAIN ---")
!export PYTHONPATH=$PYTHONPATH:$(pwd) HF_ENDPOINT=https://hf-mirror.com && python scripts/train.py

//gg colab

from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/Multimodal
# Giải nén Code từ Drive vào siêu ổ cứng SSD của Colab để tốc độ tải x10 lần
!unzip -q "/content/drive/MyDrive/Multimodal_Negative_Sampling.zip" -d "/content/Multimodal"

%cd /content/Multimodal/LVTN_2026
!pip install -r requirements.txt
!pip install nltk timm transformers
import nltk
nltk.download('punkt')

print("--- ĐANG TỰ ĐỘNG TẢI DỮ LIỆU TỪ WEB (IU-XRAY) ---")
!python scripts/prepare_dataset.py

print("--- ĐANG TẠO CÁC CỤM BỆNH LÝ (CLUSTERS) ---")
!python scripts/create_clusters.py
print("\n✅ XONG! Dữ liệu đã sẵn sàng trên SSD của Colab.")


print("--- ĐANG BẮT ĐẦU TRAIN ---")
!export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/train.py