import os
import tarfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
import json

def download_file(url, filepath, max_retries=5):
    """Download file with progress bar and retry mechanism."""
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, skipping download.")
        return
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Starting download from: {url}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=os.path.basename(filepath),
                total=total_size,
                unit='B',
                unit_scale=True,
            ) as bar:
                for data in response.iter_content(chunk_size=16384):
                    size = file.write(data)
                    bar.update(size)
            print(f"Download successful: {filepath}")
            return
            
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\n Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("No more retries. Please check your internet connection or server status.")
                raise e

def extract_tgz(filepath, extract_dir):
    """Extract .tgz files"""
    print(f"Extracting {filepath} to {extract_dir}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete!")

def parse_xml_to_csv(xml_dir, images_dir, projections_csv, output_csv):
    """Parse IU-Xray XML reports"""
    data_list = []
    projections_df = pd.read_csv(projections_csv) if os.path.exists(projections_csv) else None
    
    def normalize_id(x):
        if pd.isna(x): return x
        x = str(x).split('.')[0]
        if x.startswith('CXR'): x = x[3:]
        return x

    if projections_df is not None:
        id_col = 'id' if 'id' in projections_df.columns else 'filename'
        projections_df['match_id'] = projections_df[id_col].apply(normalize_id)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"Processing {len(xml_files)} IU-Xray XML files...")
    for xml_filename in tqdm(xml_files):
        xml_path = os.path.join(xml_dir, xml_filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            report_text = ""
            for abstract in root.findall(".//AbstractText"):
                label = abstract.get('Label')
                if label in ['FINDINGS', 'IMPRESSION'] and abstract.text:
                    report_text += abstract.text + " "
            report_text = report_text.strip()
            if not report_text: continue
                
            for parentimage in root.findall(".//parentImage"):
                img_id = parentimage.get('id')
                img_path = os.path.abspath(os.path.join(images_dir, f"{img_id}.png"))
                projection = "Unknown"
                if projections_df is not None:
                    norm_img_id = normalize_id(img_id)
                    match = projections_df[projections_df['match_id'] == norm_img_id]
                    if not match.empty: projection = match['projection'].values[0]

                data_list.append({
                    "uid": xml_filename.split('.')[0],
                    "image_id": img_id,
                    "image_path": img_path,
                    "report": report_text,
                    "projection": projection
                })
        except Exception as e: pass
            
    df = pd.DataFrame(data_list)
    valid_mask = df['image_path'].apply(os.path.exists)
    df = df[valid_mask]
    df.to_csv(output_csv, index=False)
    print(f"✅ IU-Xray Dataset created: {len(df)} samples")

def prepare_lmod_dataset(raw_dir, processed_dir):
    """
    Hỗ trợ bộ dữ liệu ODIR-5K (Tải từ Kaggle) hoặc LMOD gốc.
    """
    print("\n=== STEP 3: PREPARE OPHTHALMOLOGY DATASET (ODIR/LMOD) ===")
    lmod_raw_path = os.path.join(raw_dir, "LMOD")
    output_csv = os.path.join(processed_dir, "lmod_dataset.csv")

    if not os.path.exists(lmod_raw_path):
        print(f"⚠️ Không thấy dữ liệu tại {lmod_raw_path}")
        print("💡 Hãy chạy lệnh: !kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k --unzip -p data/raw/LMOD")
        # Tạo file mock tạm thời để code không crash
        mock_data = pd.DataFrame([{"image_id": "dummy", "image_path": "dummy.png", "report": "dummy"}])
        mock_data.to_csv(output_csv, index=False)
        return

    data = []
    # --- Ưu tiên nhận diện cấu trúc ODIR-5K ---
    odir_csv = os.path.join(lmod_raw_path, "full_df.csv")
    if os.path.exists(odir_csv):
        print("✨ Đã tìm thấy cấu trúc ODIR-5K, đang xử lý...")
        df_odir = pd.read_csv(odir_csv)
        # ODIR-5K thường có thư mục preprocessed_images
        img_dir = os.path.join(lmod_raw_path, "preprocessed_images")
        if not os.path.exists(img_dir):
            img_dir = lmod_raw_path # Thử dùng thư mục gốc
        
        for _, row in tqdm(df_odir.iterrows(), total=len(df_odir), desc="Parsing ODIR"):
            # Mắt trái
            left_img_name = row['Left-Fundus']
            left_img_path = os.path.abspath(os.path.join(img_dir, str(left_img_name)))
            if os.path.exists(left_img_path):
                data.append({
                    "image_id": left_img_name,
                    "image_path": left_img_path,
                    "report": row['Left-Diagnostic Keywords']
                })
            
            # Mắt phải
            right_img_name = row['Right-Fundus']
            right_img_path = os.path.abspath(os.path.join(img_dir, str(right_img_name)))
            if os.path.exists(right_img_path):
                data.append({
                    "image_id": right_img_name,
                    "image_path": right_img_path,
                    "report": row['Right-Diagnostic Keywords']
                })
    
    # --- Nếu không có ODIR, thử scan file ảnh chung ---
    if not data:
        print("Scanning for general images in LMOD folder...")
        for root, dirs, files in os.walk(lmod_raw_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.abspath(os.path.join(root, file))
                    data.append({
                        "image_id": file, 
                        "image_path": img_path, 
                        "report": f"Placeholder caption for {file}."
                    })

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Đã tạo dataset võng mạc thành công: {len(df)} mẫu lưu tại {output_csv}")
    else:
        print("❌ Lỗi: Không tìm thấy cặp ảnh-văn bản nào.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. IU-Xray (Giả định bạn đã tải rồi)
    IU_CSV = os.path.join(PROCESSED_DIR, "iu_xray_dataset_raw.csv")
    
    # 2. Prepare ODIR/LMOD
    prepare_lmod_dataset(RAW_DIR, PROCESSED_DIR)
