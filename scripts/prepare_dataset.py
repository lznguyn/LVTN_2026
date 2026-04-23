import os
import tarfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time

def download_file(url, filepath, max_retries=5):
    """Download file with progress bar and retry mechanism."""
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, skipping download.")
        return
    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"Starting download from: {url}")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as file, tqdm(desc=os.path.basename(filepath), total=total_size, unit='B', unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=16384):
                    size = file.write(data)
                    bar.update(size)
            return
        except Exception as e:
            print(f"\n Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1: time.sleep((attempt + 1) * 5)
            else: raise e

def parse_xml_to_csv(xml_dir, images_dir, projections_csv, output_csv):
    """Parse IU-Xray XML reports with full columns"""
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
        except Exception: pass
            
    df = pd.DataFrame(data_list)
    valid_mask = df['image_path'].apply(os.path.exists)
    df = df[valid_mask]
    df.to_csv(output_csv, index=False)
    print(f"✅ IU-Xray Dataset created with {len(df)} samples and UID column.")

def parse_kaggle_to_csv(kaggle_dir, output_csv):
    """Xử lý dữ liệu từ Kaggle nếu có sẵn"""
    print(f"🚀 Detected Kaggle dataset at: {kaggle_dir}")
    reports_path = os.path.join(kaggle_dir, "indiana_reports.csv")
    projections_path = os.path.join(kaggle_dir, "indiana_projections.csv")
    images_dir = os.path.join(kaggle_dir, "images", "images_normalized")
    if not os.path.exists(images_dir): images_dir = os.path.join(kaggle_dir, "images")
    
    if not os.path.exists(reports_path): return False
    
    reports_df = pd.read_csv(reports_path)
    projections_df = pd.read_csv(projections_path)
    reports_df['uid'] = reports_df['uid'].astype(str)
    projections_df['uid'] = projections_df['uid'].astype(str)
    
    df = pd.merge(projections_df, reports_df, on='uid', how='inner')
    df['findings'] = df['findings'].fillna('')
    df['impression'] = df['impression'].fillna('')
    df['report'] = (df['findings'] + " " + df['impression']).str.strip()
    df = df[df['report'] != '']
    df['image_path'] = df['filename'].apply(lambda x: os.path.abspath(os.path.join(images_dir, str(x))))
    df['image_id'] = df['filename'].apply(lambda x: str(x).split('.')[0])
    
    valid_df = df[df['image_path'].apply(os.path.exists)]
    output_df = valid_df[['uid', 'image_id', 'image_path', 'report', 'projection']]
    output_df.to_csv(output_csv, index=False)
    print(f"✅ Kaggle IU-Xray transformed with {len(output_df)} samples.")
    return True

def prepare_lmod_dataset(raw_dir, processed_dir):
    print("\n=== STEP 3: PREPARE OPHTHALMOLOGY DATASET (ODIR/LMOD) ===")
    lmod_raw_path = os.path.join(raw_dir, "LMOD")
    output_csv = os.path.join(processed_dir, "lmod_dataset.csv")
    if not os.path.exists(lmod_raw_path):
        print("⚠️ LMOD source not found. Run !kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k --unzip -p data/raw/LMOD")
        pd.DataFrame([{"uid":"dummy","image_id":"dummy","image_path":"dummy.png","report":"dummy","projection":"Unknown"}]).to_csv(output_csv, index=False)
        return
    
    data = []
    odir_csv = os.path.join(lmod_raw_path, "full_df.csv")
    if os.path.exists(odir_csv):
        df_odir = pd.read_csv(odir_csv)
        img_dir = os.path.join(lmod_raw_path, "preprocessed_images")
        if not os.path.exists(img_dir): img_dir = lmod_raw_path
        for _, row in df_odir.iterrows():
            for eye in ['Left', 'Right']:
                img_name = row[f'{eye}-Fundus']
                img_path = os.path.abspath(os.path.join(img_dir, str(img_name)))
                if os.path.exists(img_path):
                    data.append({"uid": str(row['ID']), "image_id": img_name, "image_path": img_path, "report": row[f'{eye}-Diagnostic Keywords'], "projection": eye})
    
    if data:
        pd.DataFrame(data).to_csv(output_csv, index=False)
        print(f"✅ ODIR Dataset created: {len(data)} samples.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(RAW_DIR, exist_ok=True); os.makedirs(PROCESSED_DIR, exist_ok=True)
    OUTPUT_CSV = os.path.join(PROCESSED_DIR, "iu_xray_dataset_raw.csv")

    # Kaggle check
    kaggle_found = False
    if os.path.exists("/kaggle/input"):
        for r, d, f in os.walk("/kaggle/input"):
            if "indiana_reports.csv" in f:
                kaggle_found = parse_kaggle_to_csv(r, OUTPUT_CSV)
                break
    
    if not kaggle_found:
        REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
        IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
        PROJECTIONS_URL = "https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university/resolve/main/indiana_projections.csv"
        
        reports_tgz = os.path.join(RAW_DIR, "NLMCXR_reports.tgz")
        images_tgz = os.path.join(RAW_DIR, "NLMCXR_png.tgz")
        projections_csv = os.path.join(RAW_DIR, "indiana_projections.csv")
        
        try:
            download_file(REPORTS_URL, reports_tgz)
            download_file(IMAGES_URL, images_tgz)
            download_file(PROJECTIONS_URL, projections_csv)
            extract_tgz(reports_tgz, os.path.join(RAW_DIR, "reports"))
            extract_tgz(images_tgz, os.path.join(RAW_DIR, "images"))
            parse_xml_to_csv(os.path.join(RAW_DIR, "reports", "ecgen-radiology"), os.path.join(RAW_DIR, "images"), projections_csv, OUTPUT_CSV)
        except Exception as e: print(f"Error: {e}")

    prepare_lmod_dataset(RAW_DIR, PROCESSED_DIR)
