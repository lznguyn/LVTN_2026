import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def create_text_clusters(input_csv, output_train_csv, output_val_csv, n_clusters=14):
    print("=== BƯỚC 1: ĐỌC DỮ LIỆU ===")
    df = pd.read_csv(input_csv)
    print(f"Tổng số mẫu tìm thấy: {len(df)}")
    
    # Lấy danh sách đoạn báo cáo Text
    reports = df['report'].fillna("").tolist()
    
    print(f"\n=== BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN (NLP EMBEDDING) ===")
    print("Đang tải khối mô hình Sentence-BERT...")
    # Dùng MiniLM (được train sẵn rất tốt) để nhúng văn bản thành Vector 384 chiều.
    # Trong tương lai bạn có thể đổi thành 'emilyalsentzer/Bio_ClinicalBERT' để thử biểu diễn Y Tế.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Đang chuyển đổi chuỗi Văn bản báo cáo thành Vector toán học (Quá trình này tốn khoảng 30s-1p)...")
    embeddings = model.encode(reports, show_progress_bar=True)
    
    print(f"\n=== BƯỚC 3: PHÂN CỤM KHÔNG GIÁM SÁT K-MEANS (K={n_clusters}) ===")
    # Phân các báo cáo thành các nhóm bệnh (Ví dụ: 14 nhóm bệnh ở phổi)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(embeddings)
    
    # Gắn nhãn cụm vừa tìm được vào lại dữ liệu bảng
    df['cluster_id'] = cluster_ids
    
    print("\nPhân bố số lượng bệnh nhân trong các Nhóm (Cluster_ID):")
    print(df['cluster_id'].value_counts().sort_index())
    
    print(f"\n=== BƯỚC 4: CHIA TẬP HUẤN LUYỆN & XÁC THỰC VÀ LƯU LẠI ===")
    # Xáo trộn ngẫu nhiên dữ liệu để Model học tốt hơn
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Chia theo tỷ lệ 80% Train : 20% Val
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)
    
    print(f"✅ Đã tạo tập Train: {len(train_df)} mẫu -> Lưu tại: {output_train_csv}")
    print(f"✅ Đã tạo tập Val:   {len(val_df)} mẫu -> Lưu tại: {output_val_csv}")
    print("\n🎉 DỮ LIỆU ĐÃ SẴN SÀNG CHO BƯỚC TRAIN MODEL!")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "iu_xray_dataset_raw.csv")
    TRAIN_CSV = os.path.join(BASE_DIR, "data", "splits", "train.csv")
    VAL_CSV = os.path.join(BASE_DIR, "data", "splits", "val.csv")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Lỗi: Không tìm thấy file {INPUT_CSV}.")
        print("Hãy chắc chắn bạn đã chạy xong hàm parse_xml_to_csv ở file prepare_dataset.py trước!")
    else:
        create_text_clusters(INPUT_CSV, TRAIN_CSV, VAL_CSV, n_clusters=14)
