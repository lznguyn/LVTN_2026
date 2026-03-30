import torch
import os

def peek_pth_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        return

    print("==================================================================")
    print(f"🔍 ĐANG KHÁM PHÁ BỘ NÃO AI: {os.path.basename(file_path)}")
    print("==================================================================")

    # Nạp trọng số (Load state dictionary)
    try:
        # Load lên CPU để xem cho nhẹ máy
        state_dict = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Lỗi khi mở file: {e}")
        return

    # 1. Thống kê tổng quan
    num_layers = len(state_dict.keys())
    total_params = sum(p.numel() for p in state_dict.values())
    
    print(f"✅ Số lượng lớp mạng (Layers): {num_layers}")
    print(f"✅ Tổng số tham số (Weights): {total_params:,} parameters")
    print("-" * 66)

    # 2. Liệt kê các lớp quan trọng nhất (Top 20 lớp đầu tiên để demo)
    print(f"{'TÊN LỚP (LAYER NAME)':<50} | {'KÍCH THƯỚC (SHAPE)':<15}")
    print("-" * 66)
    
    count = 0
    for key, value in state_dict.items():
        # Chỉ in các lớp trọng số chính (weight/bias), bỏ qua các thông số phụ
        if "weight" in key or "bias" in key:
            print(f"{key:<50} | {str(list(value.shape)):<15}")
            count += 1
        
        if count >= 30: # Chỉ in 30 lớp tiêu biểu vì Swin Transformer có hàng trăm lớp
            print("... và còn nhiều lớp khác nữa ...")
            break

    print("-" * 66)
    print("💡 MẸO: Bạn thấy các lớp có tiền tố 'image_encoder' chính là SwinV2,")
    print("      còn 'text_encoder' chính là ClinicalBERT đã được huấn luyện!")

if __name__ == "__main__":
    # Cấu hình đường dẫn file mặc định
    PATH = "checkpoints/best_model.pth"
    peek_pth_file(PATH)
