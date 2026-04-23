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
    Prepare LMOD (Ophthalmology) Dataset.
    LMOD files are typically organized by modality.
    """
    print("\n=== STEP 3: PREPARE LMOD (OPHTHALMOLOGY) DATASET ===")
    lmod_raw_path = os.path.join(raw_dir, "LMOD")
    output_csv = os.path.join(processed_dir, "lmod_dataset.csv")

    if not os.path.exists(lmod_raw_path):
        print(f"⚠️ LMOD source not found at {lmod_raw_path}")
        print("💡 Please download from https://kfzyqin.github.io/lmod/ and extract to data/raw/LMOD")
        return

    # LMOD structure often includes a master JSON or CSV for QA/Captions
    # We look for common filenames like 'lmod_captions.json' or 'metadata.csv'
    data = []
    meta_path = os.path.join(lmod_raw_path, "metadata.csv")
    
    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path)
        # Standardize columns to [image_id, image_path, report]
        # This part depends on the exact LMOD schema
        for _, row in df_meta.iterrows():
            img_rel = row.get('image_path') or row.get('filename')
            report = row.get('caption') or row.get('report') or row.get('text')
            if img_rel and report:
                img_path = os.path.abspath(os.path.join(lmod_raw_path, img_rel))
                if os.path.exists(img_path):
                    data.append({"image_id": img_rel, "image_path": img_path, "report": report})
    else:
        # Fallback: scan for images if no metadata is found (for testing)
        print("No metadata.csv found in LMOD, scanning directories...")
        for root, dirs, files in os.walk(lmod_raw_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.abspath(os.path.join(root, file))
                    data.append({
                        "image_id": file, 
                        "image_path": img_path, 
                        "report": f"Placeholder caption for {file}. Please update with real LMOD annotations."
                    })

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"✅ LMOD Dataset created: {len(df)} samples saved to {output_csv}")
    else:
        print("❌ Error: No valid image-text pairs found for LMOD.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Prepare IU-Xray
    IU_CSV = os.path.join(PROCESSED_DIR, "iu_xray_dataset_raw.csv")
    # ... (Download and parse logic as before) ...
    # (I'm keeping it concise here, assuming the user knows the IU parts are there)
    
    # 2. Prepare LMOD
    prepare_lmod_dataset(RAW_DIR, PROCESSED_DIR)
