# Clustering-Guided Multimodal Negative Sampling

Dự án triển khai ý tưởng "Lấy mẫu âm tính dựa trên phân cụm đa phương thức" cho hệ thống biểu diễn Đặc trưng (Representation Learning) đối với dataset cặp Ảnh Y Tế - Báo cáo Y tế.

## Các tính năng chính (Tasks)
1. Xử lý tập dữ liệu cặp Image-Text với thông tin Nhóm/Cụm bệnh (Cluster_ID).
2. Xây dựng Kiến trúc mạng sử dụng Swin Transformer V2 và ClinicalBERT/BioBERT, qua MLP Projection layer.
3. Huấn luyện bằng Clustering-Guided Contrastive Loss nhằm giảm thiểu hiện tượng âm tính giả (False Negatives).

## Cách chạy
1. Cài đặt các requirements
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy tiền xử lý dữ liệu và cấu hình tại `configs/default.yaml`.
3. Khởi chạy huấn luyện bằng script `scripts/train.py`.

pip install -r requirements.txt
$env:PYTHONIOENCODING="utf8"; python scripts\prepare_dataset.py
python scripts\create_clusters.py

