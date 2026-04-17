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
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# =========================================================
# Cấu hình Phân cụm Mềm (Soft Clustering)
# N_COMPONENTS: Số lượng cụm (bệnh lý)
# PCA_DIM: Giảm chiều để GMM hoạt động ổn định trên tập dữ liệu nhỏ
# =========================================================
N_COMPONENTS = 50
PCA_DIM = 32 # Giảm từ 384 -> 32 chiều

def create_text_clusters(input_csv, output_train_csv, output_val_csv, n_components=N_COMPONENTS):
    print("=" * 60)
    print("TASK 3: SOFT CLUSTERING-GUIDED NEGATIVE SAMPLING (GMM)")
    print(f"Mục tiêu: Phân {n_components} cụm xác suất bằng Gaussian Mixture")
    print("=" * 60)

    # 1. Đọc dữ liệu
    print("\n[1] Đọc dữ liệu...")
    df = pd.read_csv(input_csv)
    reports = df['report'].fillna("").tolist()

    # 2. Text Embedding với Sentence-BERT
    print("\n[2] Trích xuất đặc trưng văn bản (Sentence-BERT)...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = st_model.encode(reports, show_progress_bar=True, batch_size=64)
    
    # --- MỚI: Giảm chiều PCA để GMM ổn định hơn ---
    print(f"\n[2.1] Giảm chiều PCA (384 -> {PCA_DIM})...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"    ✅ Tổng phương sai giữ lại: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")

    # 3. Gaussian Mixture Model (Soft Clustering)
    print(f"\n[3] GMM Phân cụm mềm (n_components={n_components})...")
    gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='diag', max_iter=200)
    gmm.fit(embeddings_pca)
    
    # Lấy xác suất thuộc về từng cụm (Soft Labels)
    soft_labels = gmm.predict_proba(embeddings_pca) # (N, 50)
    # Lấy ID cụm cao nhất để tương thích với code cũ (Hard Labels)
    cluster_ids = np.argmax(soft_labels, axis=1)

    df['cluster_id'] = cluster_ids
    # Lưu index gốc để map với file .npy sau này
    df['data_idx'] = range(len(df))

    # 4. Chia Train / Val (80/20)
    print(f"\n[4] Chia tập Train/Val và lưu kết quả...")
    # Shuffle đồng bộ cả DF và ma trận Soft Labels
    indices = np.random.permutation(len(df))
    df = df.iloc[indices].reset_index(drop=True)
    soft_labels = soft_labels[indices]

    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    val_df   = df.iloc[train_size:].copy()
    
    train_soft = soft_labels[:train_size]
    val_soft   = soft_labels[train_size:]

    # Lưu file CSV cho metadata
    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv,   index=False)

    # Lưu file .npy cho Soft Labels (Xác suất)
    # File này sẽ được Dataset nạp vào khi train
    processed_dir = os.path.dirname(input_csv)
    np.save(os.path.join(processed_dir, "soft_labels_train.npy"), train_soft)
    np.save(os.path.join(processed_dir, "soft_labels_val.npy"), val_soft)

    print(f"    ✅ Train CSV: {len(train_df)} mẫu  | Soft Labels: {train_soft.shape}")
    print(f"    ✅ Val CSV:   {len(val_df)} mẫu  | Soft Labels: {val_soft.shape}")
    print(f"    📂 Soft labels saved at: {processed_dir}")

    print("\n" + "=" * 60)
    print("🎉 HOÀN TẤT! Hãy chạy train.py để huấn luyện với nhãn MỀM.")
    print("=" * 60)

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
        create_text_clusters(INPUT_CSV, TRAIN_CSV, VAL_CSV, n_components=N_COMPONENTS)
