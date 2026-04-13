"""
create_clusters.py
==================
Task 3 - Clustering-Guided Negative Sampling (Luận văn LVTN)

Mục tiêu: Phân nhóm các báo cáo y tế thành các cụm ngữ nghĩa (semantic clusters)
bằng Sentence-BERT + K-Means, sao cho các báo cáo cùng nhóm bệnh (ví dụ:
"Tăng huyết áp", "Phù phổi", "Bình thường"...) được gộp vào cùng 1 cluster_id.

Trong quá trình HUẤn LUYỆN (contrastive.py), nếu ảnh A và báo cáo B thuộc
cùng cluster → KHÔNG bị coi là mẫu âm tính (False Negative) của nhau, kể cả
khi chúng đến từ các bệnh nhân khác nhau.

Tăng n_clusters từ 14 → 50:
  - 14 cụm: mỗi cụm ~100 mẫu → quá rộng → mask sai quá nhiều negatives → Loss học kém → R@1 thấp
  - 50 cụm: mỗi cụm ~25 mẫu → phân biệt rõ hơn → loss học chính xác hơn → R@1 tăng
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# =========================================================
# Số cụm tối ưu cho IU X-Ray dataset (~7500 ảnh, ~3500 report)
# Công thức thực nghiệm: n_clusters ~ sqrt(N/2)
# Với N~3500 báo cáo → sqrt(1750) ≈ 42 → làm tròn lên 50
# =========================================================
N_CLUSTERS = 50

def create_text_clusters(input_csv, output_train_csv, output_val_csv, n_clusters=N_CLUSTERS):
    print("=" * 60)
    print("TASK 3: CLUSTERING-GUIDED NEGATIVE SAMPLING")
    print(f"Mục tiêu: Phân {n_clusters} cụm ngữ nghĩa báo cáo y tế")
    print("=" * 60)

    # ─────────────────────────────────────────────
    # BƯỚC 1: Đọc dữ liệu đầu vào
    # ─────────────────────────────────────────────
    print("\n[1] Đọc dữ liệu...")
    df = pd.read_csv(input_csv)
    print(f"    Tổng số mẫu: {len(df)}")

    reports = df['report'].fillna("").tolist()

    # ─────────────────────────────────────────────
    # BƯỚC 2: Text Embedding với Sentence-BERT
    # (Dùng MiniLM vì nhẹ, nhanh, độ chính xác tốt)
    # Có thể thay bằng Bio_ClinicalBERT để cải thiện
    # ─────────────────────────────────────────────
    print("\n[2] Trích xuất đặc trưng văn bản (Sentence-BERT)...")
    print("    Đang tải mô hình all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("    Đang encode toàn bộ báo cáo thành vector 384 chiều...")
    embeddings = model.encode(reports, show_progress_bar=True, batch_size=64)
    print(f"    ✅ Đặc trưng thu được: shape = {embeddings.shape}")

    # ─────────────────────────────────────────────
    # BƯỚC 3: K-Means Clustering
    # Phân báo cáo thành n_clusters nhóm bệnh ngữ nghĩa
    # ─────────────────────────────────────────────
    print(f"\n[3] K-Means phân cụm (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', max_iter=500)
    cluster_ids = kmeans.fit_predict(embeddings)

    df['cluster_id'] = cluster_ids

    # Phân tích phân phối: mỗi cụm có bao nhiêu mẫu?
    sizes = df['cluster_id'].value_counts().sort_index()
    print(f"\n    Phân phối số mẫu/cụm:")
    print(f"    Min: {sizes.min()} | Max: {sizes.max()} | TB: {sizes.mean():.1f} mẫu/cụm")
    print(f"\n    TOP 10 cụm lớn nhất:")
    print(sizes.head(10).to_string())

    # ─────────────────────────────────────────────
    # BƯỚC 4: Chia Train / Val (80/20), lưu CSV
    # ─────────────────────────────────────────────
    print(f"\n[4] Chia tập Train/Val và lưu kết quả...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    val_df   = df.iloc[train_size:].copy()

    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv,   index=False)

    print(f"    ✅ Train: {len(train_df)} mẫu  → {output_train_csv}")
    print(f"    ✅ Val:   {len(val_df)} mẫu  → {output_val_csv}")

    print("\n" + "=" * 60)
    print("🎉 HOÀN TẤT! Dữ liệu sẵn sàng cho TRAINING.")
    print("   Hãy chạy: python scripts/train.py")
    print("=" * 60)


if __name__ == "__main__":
    BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    INPUT_CSV   = os.path.join(BASE_DIR, "data", "processed", "iu_xray_dataset_raw.csv")
    TRAIN_CSV   = os.path.join(BASE_DIR, "data", "splits",    "train.csv")
    VAL_CSV     = os.path.join(BASE_DIR, "data", "splits",    "val.csv")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Không tìm thấy: {INPUT_CSV}")
        print("   → Hãy chạy prepare_dataset.py trước!")
    else:
        create_text_clusters(INPUT_CSV, TRAIN_CSV, VAL_CSV, n_clusters=N_CLUSTERS)
