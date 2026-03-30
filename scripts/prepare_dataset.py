import os
import tarfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

def download_file(url, filepath):
    """Tải file có hiển thị Thanh tiến trình (Progress bar)."""
    if os.path.exists(filepath):
        print(f"File {filepath} đã tồn tại, bỏ qua bước tải mới.")
        return
    print(f"Bắt đầu tải từ: {url}")
    
    # Sử dụng requests để tránh lỗi SSL Handshake của urllib
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.split('\\')[-1],
        total=total_size,
        unit='B',
        unit_scale=True,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tgz(filepath, extract_dir):
    """Giải nén file nén .tgz Linux"""
    print(f"Đang giải nén {filepath} vào {extract_dir}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Giải nén hoàn tất!")

def parse_xml_to_csv(xml_dir, images_dir, output_csv):
    """Phân tích các báo cáo XML, trích xuất text và ráp với ID của file ảnh"""
    data_list = []
    
    # Lấy danh sách file xml (Báo cáo khám bệnh)
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if len(xml_files) == 0:
        print(f"Cảnh báo: Không tìm thấy file XML nào trong thư mục {xml_dir}.")
        return

    print(f"Đang xử lý {len(xml_files)} file báo cáo XML để tạo thành CSDL...")
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
                
                # Tạo bản ghi ghép giữa Văn bản và Đường dẫn ảnh cụ thể
                data_list.append({
                    "image_path": img_path,
                    "report": report_text,
                })
        except Exception as e:
            print(f"Lỗi khi xử lý {xml_filename}: {str(e)}")
            
    # Chuyển đổi thành dạng Bảng (Dataframe)
    df = pd.DataFrame(data_list)
    
    # Làm sạch: Chỉ giữ lại các hàng có file ảnh tồn tại thật trên ổ cứng
    valid_mask = df['image_path'].apply(os.path.exists)
    df = df[valid_mask]
    
    # Lưu ra định dạng CSV để huấn luyện
    df.to_csv(output_csv, index=False)
    print(f"✅ Đã tạo CSDL thành công! Tổng cộng: {len(df)} mẫu hợp lệ.")
    print(f"File lưu tại: {output_csv}")
    print("Bạn có thể mở file CSV bằng pandas để xem thử cột image_path và report.")

if __name__ == "__main__":
    # KHAI BÁO CÁC ĐƯỜNG DẪN TƯƠNG ĐỐI DỰA TRÊN CẤU TRÚC FOLDER
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # --- THÔNG TIN NGUỒN TẢI ---
    # Nguồn: Indiana University Chest X-Rays (Khoảng ~4000 báo cáo và ~7500 bức ảnh)
    REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
    IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
    
    reports_tgz = os.path.join(RAW_DATA_DIR, "NLMCXR_reports.tgz")
    images_tgz = os.path.join(RAW_DATA_DIR, "NLMCXR_png.tgz")
    reports_unzip_dir = os.path.join(RAW_DATA_DIR, "reports")
    images_unzip_dir = os.path.join(RAW_DATA_DIR, "images")
    
    # =========================================================================
    # LƯU Ý QUAN TRỌNG VỀ TẢI DỮ LIỆU:
    # BỎ COMMENT (xóa dấu #) ở khối code dưới đây nếu bạn muốn TẢI TRỰC TIẾP
    # Thời gian tải có thể mất từ 10 - 20 phút (File ảnh nặng khoảng vài GB).
    # =========================================================================
    
    print("=== BƯỚC 1: TẢI VÀ GIẢI NÉN (Có thể mở file script để kích hoạt nếu chưa tải) ===")
    
    os.makedirs(reports_unzip_dir, exist_ok=True)
    os.makedirs(images_unzip_dir, exist_ok=True)
    
    download_file(REPORTS_URL, reports_tgz)
    extract_tgz(reports_tgz, reports_unzip_dir)
    
    download_file(IMAGES_URL, images_tgz)
    extract_tgz(images_tgz, images_unzip_dir)
    
    print("\n=== BƯỚC 2: BUILD DATASET PIPELINE ===")
    output_csv = os.path.join(PROCESSED_DATA_DIR, "iu_xray_dataset_raw.csv")
    # Báo cáo XML thực chất được giải nén ra một thư mục con
    actual_xml_dir = os.path.join(reports_unzip_dir, "ecgen-radiology")
    parse_xml_to_csv(actual_xml_dir, images_unzip_dir, output_csv)
