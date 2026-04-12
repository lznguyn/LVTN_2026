import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

import sys
# Gắn thư mục gốc vào đường dẫn hệ thống để dễ gọi các modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.data.dataset import MedicalImageTextDataset
from src.engine.trainer import MultimodalTrainer
from scripts.evaluate import evaluate_retrieval

def load_config(config_path="configs/default.yaml"):
    """Đọc file tùy chỉnh siêu tham số YAML"""
    with open(config_path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)

def get_transforms(image_size):
    """Ảnh ImageNet được Normalize theo tiêu chuẩn thế giới"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    print("==================================================================")
    print("🚀 LUẬN VĂN: HUẤN LUYỆN MULTIMODAL CLUSTERING-GUIDED NEGATIVE SAMPLING 🚀")
    print("==================================================================")
    
    # Kích hoạt sử dụng GPU để train Nhanh hơn (Nếu máy bạn có Card rời NVIDIA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 AI đang tính toán bằng lõi: {device.upper()}")
    
    config = load_config()
    
    # 1. GIAI ĐOẠN DATASET
    print("\n[1] Đang nạp bộ từ điển Tokenizer của mô hình ClinicalBERT...")
    # Tải Tokenizer chuyên dụng cho dữ liệu sinh khoa/y tế
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    image_transform = get_transforms(config['data']['image_size'])
    
    print("[2] Đang Load Database...")
    try:
        train_df = pd.read_csv(config['data']['train_csv'])
        val_df = pd.read_csv(config['data']['val_csv'])
    except Exception as e:
        print(f"❌ Không tìm thấy file. Lỗi: {e}")
        print("💡 Chú ý: Hãy phải chạy Python trên 2 file prepare_dataset.py và create_clusters.py trước!")
        return

    train_dataset = MedicalImageTextDataset(train_df, image_transform, tokenizer, config['data']['max_text_length'])
    val_dataset = MedicalImageTextDataset(val_df, image_transform, tokenizer, config['data']['max_text_length'])
    
    # Sửa lỗi Threading/Crash của hệ điều hành Windows khi gọi num_workers > 0
    num_workers = 0 if os.name == 'nt' else config['training']['num_workers']
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
    
    # 2. GIAI ĐOẠN KHỞI TẠO NEURAL NETWORK
    print("\n[3] 🏗️ Đang xây dựng cấu trúc Mạng Neuron (Swin Transformer V2 + ClinicalBERT + MLP)...")
    model = MultimodalModel(
        image_encoder_name=config['model']['image_encoder'],
        text_model_name=config['model']['text_encoder'],
        embed_dim=config['model']['embed_dim']
    )
    
    print(f"Tổng số tham số mạng lưu giữ: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} Parameters")
    
    # 3. GIAI ĐOẠN TIẾN HÀNH TRAINING LOOP
    print("\n[4] ⚡ TIẾN HÀNH ĐÀO TẠO MÔ HÌNH VỚI DỮ LIỆU...")
    trainer = MultimodalTrainer(model, config, device=device)
    
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    
    # Tạo thư mục chứa tệp file .pth Weights cuối cùng
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)
        
        # --- THÊM PHẦN ĐÁNH GIÁ R@1 NGAY TẠI ĐÂY ---
        print(f"\n📊 Đang đánh giá chỉ số R@1 (Retrieval) cho Epoch {epoch}...")
        try:
            i2t, t2i = evaluate_retrieval(model, val_loader, device)
            print(f"✅ Epoch {epoch} - R@1: {i2t[0]:.2f}% | R@5: {i2t[1]:.2f}% | R@10: {i2t[2]:.2f}%")
        except Exception as e:
            print(f"⚠️ Lỗi khi đánh giá R@1: {e}")
            
        # Nhớ lại mô hình "chất lượng Nhất" để lưu qua từng vòng
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(config['training']['checkpoint_dir'], "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"⭐ [CÓ CẢI THIỆN] - Đã chép file kết quả ghi đè vào: {ckpt_path}")
            
            # --- TỰ ĐỘNG BACKUP VÀO BÊN TRONG GOOGLE DRIVE ---
            drive_dir = "/content/drive/MyDrive/Multimodal_Checkpoints"
            if os.path.exists("/content/drive/MyDrive"):
                import shutil
                os.makedirs(drive_dir, exist_ok=True)
                drive_path = os.path.join(drive_dir, "best_model.pth")
                shutil.copy(ckpt_path, drive_path)
                print(f"✅ Đã backup file best_model.pth an toàn vào Google Drive: {drive_path}")
            # ---------------------------------------------------
            
    print("\n🎉 HOÀN TẤT QUÁ TRÌNH THỰC THI KHÓA LUẬN!")

if __name__ == "__main__":
    main()
