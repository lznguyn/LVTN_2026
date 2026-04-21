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
                for data in response.iter_content(chunk_size=16384): # Tăng chunk_size để tải nhanh hơn
                    size = file.write(data)
                    bar.update(size)
            print(f"Download successful: {filepath}")
            return
            
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\n⚠️ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ No more retries. Please check your internet connection or server status.")
                raise e

def extract_tgz(filepath, extract_dir):
    """Extract .tgz files"""
    print(f"Extracting {filepath} to {extract_dir}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete!")

def parse_xml_to_csv(xml_dir, images_dir, projections_csv, output_csv):
    """Parse XML reports, extract text, and map with image IDs and Projection metadata"""
    data_list = []
    
    # Load projection metadata if available
    projections_df = None
    if os.path.exists(projections_csv):
        print(f"📖 Loading metadata from: {projections_csv}")
        projections_df = pd.read_csv(projections_csv)
        
        # Determine the ID column (could be 'id' or 'filename')
        id_col = 'id' if 'id' in projections_df.columns else 'filename'
        
        # Create a standardized 'match_id' for robust matching
        # Removes extensions like .dcm.png and prefixes like CXR
        def normalize_id(x):
            if pd.isna(x): return x
            x = str(x).split('.')[0] # Remove extensions
            if x.startswith('CXR'): x = x[3:] # Remove CXR prefix if present
            return x

        projections_df['match_id'] = projections_df[id_col].apply(normalize_id)
    else:
        print(f"⚠️ Warning: Projections file {projections_csv} not found. 'projection' column will be empty.")

    # Get list of xml files
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if len(xml_files) == 0:
        print(f"Warning: No XML files found in {xml_dir}.")
        return

    print(f"Processing {len(xml_files)} XML report files...")
    for xml_filename in tqdm(xml_files):
        xml_path = os.path.join(xml_dir, xml_filename)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 1. Trích xuất Text (Thường nằm ở mục FINDINGS (Chi tiết khám) hoặc IMPRESSION (Kết luận))
            report_text = ""
            for abstract in root.findall(".//AbstractText"):
                label = abstract.get('Label')
                text = abstract.text
                if label in ['FINDINGS', 'IMPRESSION'] and text is not None:
                    report_text += text + " "
            report_text = report_text.strip()
            
            # Bỏ qua nếu bác sĩ không ghi nhận xét
            if not report_text:
                continue
                
            # 2. Lấy danh sách ảnh X-Ray đính kèm (Lưu ý: 1 báo cáo có thể chụp 2-3 tấm ảnh ở các góc độ)
            for parentimage in root.findall(".//parentImage"):
                img_id = parentimage.get('id')
                img_path = os.path.abspath(os.path.join(images_dir, f"{img_id}.png"))
                
                # Tìm thông tin projection (Frontal/Lateral)
                projection = "Unknown"
                if projections_df is not None:
                    # Normalize img_id for matching (removing CXR prefix if present)
                    norm_img_id = str(img_id).split('.')[0]
                    if norm_img_id.startswith('CXR'): norm_img_id = norm_img_id[3:]
                    
                    match = projections_df[projections_df['match_id'] == norm_img_id]
                    if not match.empty:
                        projection = match['projection'].values[0]

                # Tạo bản ghi ghép giữa Văn bản, Đường dẫn ảnh và View Type
                data_list.append({
                    "uid": xml_filename.split('.')[0], # Thêm Patient ID từ tên file XML
                    "image_id": img_id,
                    "image_path": img_path,
                    "report": report_text,
                    "projection": projection
                })
        except Exception as e:
            print(f"Error processing {xml_filename}: {str(e)}")
            
    # Chuyển đổi thành dạng Bảng (Dataframe)
    df = pd.DataFrame(data_list)
    
    # Làm sạch: Chỉ giữ lại các hàng có file ảnh tồn tại thật trên ổ cứng
    valid_mask = df['image_path'].apply(os.path.exists)
    df = df[valid_mask]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset created successfully! Total valid samples: {len(df)}")
    print(f"File saved at: {output_csv}")
    print(f"📊 View Type Distribution:\n{df['projection'].value_counts()}")
    print("You can open the CSV file using pandas to check the columns.")

def parse_kaggle_to_csv(kaggle_dir, output_csv):
    """Xử lý dữ liệu từ bộ Kaggle raddar/chest-xrays-indiana-university"""
    print(f"🚀 Detected Kaggle dataset at: {kaggle_dir}")
    
    reports_path = os.path.join(kaggle_dir, "indiana_reports.csv")
    projections_path = os.path.join(kaggle_dir, "indiana_projections.csv")
    images_dir = os.path.join(kaggle_dir, "images", "images_normalized") # raddar structure
    
    # Một số version Kaggle để folder images trực tiếp
    if not os.path.exists(images_dir):
        images_dir = os.path.join(kaggle_dir, "images")

    if not os.path.exists(reports_path) or not os.path.exists(projections_path):
        print(f"❌ Error: Missing CSV files in {kaggle_dir}")
        return False

    print("📖 Reading Kaggle CSV files...")
    reports_df = pd.read_csv(reports_path)
    projections_df = pd.read_csv(projections_path)

    # Convert uid to string for robust merging
    reports_df['uid'] = reports_df['uid'].astype(str)
    projections_df['uid'] = projections_df['uid'].astype(str)

    # Merge reports and projections
    df = pd.merge(projections_df, reports_df, on='uid', how='inner')

    # Gộp findings và impression thành report (giống logic parse XML)
    df['findings'] = df['findings'].fillna('')
    df['impression'] = df['impression'].fillna('')
    df['report'] = df['findings'] + " " + df['impression']
    df['report'] = df['report'].str.strip()

    # Chỉ giữ các dòng có báo cáo
    df = df[df['report'] != '']

    # Xây dựng đường dẫn ảnh tuyệt đối
    # Kaggle filenames trong indiana_projections thường là '1_IM-0001-4001.dcm.png'
    df['image_path'] = df['filename'].apply(lambda x: os.path.abspath(os.path.join(images_dir, str(x))))
    df['image_id'] = df['filename'].apply(lambda x: str(x).split('.')[0])

    # Chỉ giữ lại các hàng có file ảnh thực sự tồn tại
    initial_count = len(df)
    valid_mask = df['image_path'].apply(os.path.exists)
    df = df[valid_mask]
    
    if len(df) == 0:
        print(f"❌ Warning: No images found at {images_dir}. Check the folder structure.")
        return False

    print(f"✅ Filtered {len(df)}/{initial_count} valid image-report pairs.")

    # Chọn lọc các cột cần thiết để đồng bộ với pipeline hiện tại
    output_df = df[['uid', 'image_id', 'image_path', 'report', 'projection']]
    
    # Giữ toàn bộ ảnh (Frontal + Lateral) để tối đa số mẫu training
    print(f"🖼️  Giữ toàn bộ {len(output_df)} ảnh (Frontal + Lateral).")
    
    output_df.to_csv(output_csv, index=False)
    
    print(f"✅ Kaggle Dataset transformed successfully! Saved to: {output_csv}")
    print(f"📊 View Type Distribution:\n{output_df['projection'].value_counts()}")
    return True

if __name__ == "__main__":
    # KHAI BÁO CÁC ĐƯỜNG DẪN TƯƠNG ĐỐI DỰA TRÊN CẤU TRÚC FOLDER
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    output_csv = os.path.join(PROCESSED_DATA_DIR, "iu_xray_dataset_raw.csv")

    # --- TỰ ĐỘNG TÌM KIẾM DỮ LIỆU KAGGLE ---
    kaggle_base = "/kaggle/input"
    detected_kaggle_dir = None
    
    if os.path.exists(kaggle_base):
        print(f"🔍 Đang quét dữ liệu trong {kaggle_base}...")
        # Quét các thư mục trong /kaggle/input để tìm file indiana_reports.csv
        for root_dir, dirs, files in os.walk(kaggle_base):
            if "indiana_reports.csv" in files:
                detected_kaggle_dir = root_dir
                print(f"✨ Tìm thấy dữ liệu tại: {detected_kaggle_dir}")
                break
    
    if detected_kaggle_dir:
        success = parse_kaggle_to_csv(detected_kaggle_dir, output_csv)
        if success:
            exit(0)
        else:
            print("Warning: Error processing Kaggle data, switching to manual download...")
    else:
        print("Searching for dataset in /kaggle/input... No Kaggle data found. Switching to internet download...")

    # --- DOWNLOAD SOURCES ---
    # Nguồn: Indiana University Chest X-Rays (Khoảng ~4000 báo cáo và ~7500 bức ảnh)
    REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
    IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
    # Metadata bổ sung cho Frontal/Lateral views
    PROJECTIONS_URL = "https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university/resolve/main/indiana_projections.csv"
    
    reports_tgz = os.path.join(RAW_DATA_DIR, "NLMCXR_reports.tgz")
    images_tgz = os.path.join(RAW_DATA_DIR, "NLMCXR_png.tgz")
    projections_csv = os.path.join(RAW_DATA_DIR, "indiana_projections.csv")
    
    reports_unzip_dir = os.path.join(RAW_DATA_DIR, "reports")
    images_unzip_dir = os.path.join(RAW_DATA_DIR, "images")
    
    print("=== STEP 1: DOWNLOAD AND EXTRACT (Open file to activate if not downloaded) ===")
    
    os.makedirs(reports_unzip_dir, exist_ok=True)
    os.makedirs(images_unzip_dir, exist_ok=True)
    
    try:
        download_file(REPORTS_URL, reports_tgz)
        extract_tgz(reports_tgz, reports_unzip_dir)
        
        download_file(IMAGES_URL, images_tgz)
        extract_tgz(images_tgz, images_unzip_dir)
        
        # Tải thêm file metadata projection
        download_file(PROJECTIONS_URL, projections_csv)
        
        print("\n=== STEP 2: BUILD DATASET PIPELINE ===")
        actual_xml_dir = os.path.join(reports_unzip_dir, "ecgen-radiology")
        parse_xml_to_csv(actual_xml_dir, images_unzip_dir, projections_csv, output_csv)
    except Exception as e:
        print(f"❌ Failed to prepare dataset from scratch: {e}")
        print("Tip: If you are on Kaggle, please 'Add Data' the 'chest-xrays-indiana-university' dataset.")
