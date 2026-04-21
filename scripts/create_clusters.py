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
N_COMPONENTS = 20 # Giảm từ 50 xuống 20 để tăng tính đặc trưng cho mỗi cụm
PCA_DIM = 32 # Giảm từ 384 -> 32 chiều

def clean_report(text):
    """Tiền xử lý văn bản y tế để loại bỏ nhiễu"""
    import re
    if not isinstance(text, str): return ""
    text = text.lower()
    # Loại bỏ các từ vô nghĩa hệ thống
    noise = [
        r'xxxx', r'comparison', r'findings', r'indication', r'history', 
        r'clinical', r'study', r'exam', r'patient', r'chest', r'radiograph',
        r'view', r'provided', r'none', r'referred', r'available'
    ]
    for n in noise:
        text = re.sub(n, '', text)
    
    # Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_text_clusters(input_csv, output_train_csv, output_val_csv, n_components=N_COMPONENTS):
    print("=" * 60)
    print("TASK 3: OPTIMIZED SOFT CLUSTERING (GMM)")
    print(f"Goal: Clustering {n_components} semantic groups using Gaussian Mixture")
    print("=" * 60)

    # 1. Read Data
    print("\n[1] Reading and preprocessing data...")
    df = pd.read_csv(input_csv)
    reports_raw = df['report'].fillna("").tolist()
    
    # Apply cleaning
    reports_clean = [clean_report(r) for r in reports_raw]
    
    # 2. Text Embedding with Sentence-BERT
    print("\n[2] Extracting text features (Sentence-BERT)...")
    # Use 'all-mpnet-base-v2' (better for medical than MiniLM)
    st_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = st_model.encode(reports_clean, show_progress_bar=True, batch_size=32)
    
    # --- PCA: Dimension Reduction ---
    print(f"\n[2.1] Reducing dimensions with PCA (768 -> {PCA_DIM})...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"    Total variance explained: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")

    # 3. Gaussian Mixture Model (Soft Clustering)
    print(f"\n[3] GMM Soft Clustering (n_components={n_components})...")
    gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='diag', max_iter=300)
    gmm.fit(embeddings_pca)
    
    # Soft Labels (Probabilities)
    soft_labels = gmm.predict_proba(embeddings_pca) # (N, 20)
    # Get highest cluster ID for compatibility (Hard Labels)
    cluster_ids = np.argmax(soft_labels, axis=1)

    df['cluster_id'] = cluster_ids
    # Lưu index gốc để map với file .npy sau này
    df['data_idx'] = range(len(df))

    # 4. Chia Train / Val theo Patient ID (uid) Tránh Data Leakage
    print(f"\n[4] Chia tập Train/Val theo Patient-level (Trường hợp 1 ID có Frontal/Lateral)...")
    
    unique_patients = df['uid'].unique()
    np.random.seed(42) # Giữ kết quả cố định
    np.random.shuffle(unique_patients)
    
    train_size = int(0.8 * len(unique_patients))
    train_uids = unique_patients[:train_size]
    val_uids = unique_patients[train_size:]
    
    # Map ngược lại để lấy toàn bộ ảnh của các bệnh nhân trong danh sách
    train_df = df[df['uid'].isin(train_uids)].copy()
    val_df   = df[df['uid'].isin(val_uids)].copy()
    
    # Sắp xếp lại soft_labels theo đúng thứ tự của df sau khi chia
    # Vì chúng ta cần map soft_labels (xuất phát từ toàn bộ dataset) sang các tập con
    train_soft = soft_labels[train_df['data_idx'].values]
    val_soft   = soft_labels[val_df['data_idx'].values]

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
