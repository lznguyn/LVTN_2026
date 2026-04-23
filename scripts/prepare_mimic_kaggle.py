import os
import zipfile
import pandas as pd
from tqdm import tqdm
import shutil

# --- CẤU HÌNH ĐƯỜNG DẪN KAGGLE ---
TEMP_EXTRACT_DIR = "/kaggle/working/mimic_temp"
OUTPUT_DIR = "/kaggle/working/LVTN_2026/data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mimic_dataset.csv")

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        return False
    print(f"📦 Đang giải nén {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return True

def find_mimic_files():
    """Tìm kiếm file ZIP trong các dataset đã add vào Kaggle"""
    # Lấy đường dẫn gốc của project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    search_dirs = [
        "/kaggle/input", 
        os.path.join(base_dir, "data", "raw"),
        "data/raw" # Thử đường dẫn tương đối
    ]
    
    files_found = {}
    target_files = {
        "images": "MIMIC_Processed_512.zip",
        "reports": "mimic-cxr-reports.zip",
        "metadata": "mimic-cxr-2.0.0-metadata.csv.gz"
    }
    
    print(f"🔍 Đang tìm kiếm file trong: {search_dirs}")
    for base in search_dirs:
        if not os.path.exists(base): continue
        for root, dirs, filenames in os.walk(base):
            for f in filenames:
                for key, target in target_files.items():
                    if f == target:
                        files_found[key] = os.path.join(root, f)
                        print(f"✨ Tìm thấy {key}: {files_found[key]}")
    return files_found

def main():
    print("🚀 TIỀN XỬ LÝ MIMIC-CXR TRÊN KAGGLE 🚀")
    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tìm file
    files = find_mimic_files()
    if len(files) < 3:
        print("❌ Lỗi: Thiếu file dữ liệu!")
        print(f"Đã tìm thấy: {list(files.keys())}")
        print("💡 Hãy đảm bảo bạn đã add dataset chứa các file ZIP và Metadata vào Kaggle.")
        return

    # Giải nén
    img_dir = os.path.join(TEMP_EXTRACT_DIR, "images")
    rep_dir = os.path.join(TEMP_EXTRACT_DIR, "reports")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    if not os.listdir(img_dir): extract_zip(files['images'], img_dir)
    if not os.listdir(rep_dir): extract_zip(files['reports'], rep_dir)

    # Xử lý Metadata
    print("\n--- Đang xử lý Metadata ---")
    df_meta = pd.read_csv(files['metadata'])
    
    data = []
    # Lấy 5000 mẫu để đánh giá nhanh (giống logic cũ)
    df_subset = df_meta.head(5000)
    
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        dicom_id = str(row['dicom_id'])
        study_id = str(row['study_id'])
        
        # Tìm ảnh
        img_path = ""
        for ext in ['.png', '.jpg']:
            p = os.path.join(img_dir, f"{dicom_id}{ext}")
            if os.path.exists(p):
                img_path = os.path.abspath(p)
                break
        
        # Tìm report
        report_text = ""
        p_rep = os.path.join(rep_dir, f"s{study_id}.txt")
        if os.path.exists(p_rep):
            with open(p_rep, 'r') as f:
                report_text = f.read().strip()
        
        if img_path and report_text:
            data.append({
                "uid": str(row['subject_id']),
                "image_id": dicom_id,
                "image_path": img_path,
                "report": report_text,
                "projection": str(row.get('ViewPosition', 'Unknown'))
            })

    if data:
        pd.DataFrame(data).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Đã tạo file: {OUTPUT_CSV} ({len(data)} samples)")
    else:
        print("❌ Lỗi: Không ghép nối được dữ liệu ảnh và text.")

if __name__ == "__main__":
    main()
