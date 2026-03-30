import os
import zipfile
import pandas as pd
import re
from tqdm import tqdm
import shutil

# LƯU Ý KHI CHẠY TRÊN COLAB:
# Trên Google Colab, khi bạn chạy lệnh này, hãy truyền tham số thư mục gốc của Drive. 
# Mặc định file sẽ lấy đường dẫn cố định dưới đây.
DRIVE_DIR = "/content/drive/MyDrive/db_MIMIC_CXRCXR"
TEMP_EXTRACT_DIR = "/content/mimic_temp"

# Đầu ra cuối cùng sẽ nằm chung với dự án của bạn để ăn khớp với các script cũ.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def extract_zip(zip_path, extract_to):
    print(f"Bung nén {zip_path} -> {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def clean_report_text(text):
    """Trích xuất trọng tâm Y tế từ báo cáo gốc của MIMIC"""
    # Tìm nhanh phần FINDINGS và IMPRESSION
    findings = re.search(r'FINDINGS:(.*?)(?=(?:[A-Z\s]+:|$))', text, re.DOTALL | re.IGNORECASE)
    impression = re.search(r'IMPRESSION:(.*?)(?=(?:[A-Z\s]+:|$))', text, re.DOTALL | re.IGNORECASE)
    
    result = ""
    if findings: result += findings.group(1).strip() + " "
    if impression: result += impression.group(1).strip()
    
    result = " ".join(result.split())
    # Nếu file quá ngắn hoặc không có tag chuẩn, lấy toàn bộ text (cắt bớt rác đầu file)
    if len(result) < 10:
        result = " ".join(text.split())
    return result

def prepare_mimic_data():
    print("==================================================================")
    print("🚀 TIỀN XỬ LÝ DỮ LIỆU ĐA PHƯƠNG THỨC MIMIC-CXR TRÊN COLAB 🚀")
    print("==================================================================")
    
    if not os.path.exists(DRIVE_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục {DRIVE_DIR}!")
        print("💡 Gợi ý: Trên Google Drive, bạn hãy chuột phải vào 'db_MIMIC_CXRCXR' -> Chọn TỔ CHỨC -> Chọn THÊM LỐI TẮT vào Drive của tôi nhé!")
        return
    
    images_zip = os.path.join(DRIVE_DIR, "MIMIC_Processed_512.zip")
    reports_zip = os.path.join(DRIVE_DIR, "mimic-cxr-reports.zip")
    metadata_csv = os.path.join(DRIVE_DIR, "mimic-cxr-2.0.0-metadata.csv.gz")
    
    # --- BƯỚC 1: GIẢI NÉN DATA LÊN Ổ SSD CỦA COLAB ĐỂ CHẠY CHO NHANH ---
    images_unzip_dir = os.path.join(TEMP_EXTRACT_DIR, "images")
    reports_unzip_dir = os.path.join(TEMP_EXTRACT_DIR, "reports")
    
    os.makedirs(images_unzip_dir, exist_ok=True)
    os.makedirs(reports_unzip_dir, exist_ok=True)
    
    # Chỉ giải nén nếu chưa có (Tiết kiệm thời gian trên Colab nếu lỡ ngắt kết nối)
    if len(os.listdir(images_unzip_dir)) == 0:
        extract_zip(images_zip, images_unzip_dir)
    if len(os.listdir(reports_unzip_dir)) == 0:
        extract_zip(reports_zip, reports_unzip_dir)
        
    print("\n--- BƯỚC 2: NẠP METADATA CSV ---")
    print(f"Đọc {metadata_csv}...")
    df_meta = pd.read_csv(metadata_csv)
    # Lọc ra các cột thiết yếu: dicom_id (ảnh), study_id (báo cáo text)
    df_meta = df_meta[['dicom_id', 'study_id', 'subject_id']]
    
    print("\n--- BƯỚC 3: QUÉT FILE ẢNH VÀ TEXT ---")
    # Quét tất cả file ảnh đã giải nén
    print("Đang quét cấu trúc file Ảnh...")
    image_paths = {}
    for root, _, files in os.walk(images_unzip_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                # File ảnh thường có tên dạng dicom_id.jpg
                dicom_id = os.path.splitext(file)[0]
                image_paths[dicom_id] = os.path.join(root, file)
                
    # Quét tất cả file báo cáo đã giải nén
    print("Đang quét cấu trúc file Báo cáo...")
    report_texts = {}
    for root, _, files in os.walk(reports_unzip_dir):
        for file in tqdm(files, desc="Đang phân tích Ngữ Nghĩa Text"):
            if file.endswith(".txt"):
                study_id = os.path.splitext(file)[0]
                # Ở MIMIC, file text hay có tiền tố 's' ví dụ 's50414267.txt'
                if study_id.startswith('s'):
                    study_id = study_id[1:]
                try:    
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                        report_texts[int(study_id)] = clean_report_text(text)
                except Exception:
                    pass

    print("\n--- BƯỚC 4: LẮP RÁP (JOIN) ẢNH VÀ BÁO CÁO THÀNH DATASET CHUẨN ---")
    # Giữ nguyên logic của hệ thống cũ: Yêu cầu file CSV cột 1: 'image_path', cột 2: 'report'
    dataset_records = []
    
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Gắn kết Đa phương thức"):
        dicom_id = str(row['dicom_id'])
        study_id = int(row['study_id'])
        
        # Chỉ lấy cặp nào tồn tại cả Ảnh vật lý và Báo cáo vật lý
        if dicom_id in image_paths and study_id in report_texts:
            dataset_records.append({
                'image_path': image_paths[dicom_id],
                'report': report_texts[study_id]
            })
            
    df_final = pd.DataFrame(dataset_records)
    print(f"\n✅ Đã ghép thành công {len(df_final)} cặp Dữ liệu!")
    
    # LƯU Ý CỰC KỲ QUAN TRỌNG ĐỂ KHÔNG PHÁ VỠ LOGIC CŨ LVTN CỦA BẠN:
    # Tôi sẽ cố tình lưu tên file giống hệt cái file cũ của OpenI (iu_xray_dataset_raw.csv).
    # Nhờ mẹo này, hàm create_clusters.py và hàm train.py của bạn SẼ KHÔNG CẦN CHỈNH SỬA BẤT KỲ CHỮ NÀO mà vẫn chạy trơn tru với dữ liệu MIMIC!
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_csv = os.path.join(PROCESSED_DATA_DIR, "iu_xray_dataset_raw.csv") # Cố tình đặt tên cũ
    df_final.to_csv(out_csv, index=False)
    print(f"Đã xuất file Dữ Liệu Gốc Siêu Cấp tại: {out_csv}")
    print("\n🎉 NHIỆM VỤ ĐÃ HOÀN TẤT! GIỜ BẠN CHỈ CẦN CHẠY KẾT TIẾP LỆNH: !python scripts/create_clusters.py")

if __name__ == "__main__":
    prepare_mimic_data()
