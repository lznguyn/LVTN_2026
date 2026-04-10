import torch
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import MedicalImageTextDataset
from src.models.hrgr_agent import HRGRAgent
from src.engine.rl_trainer import HRGRRLTrainer
from src.data.vocabulary import WordVocabulary
import json
from transformers import AutoTokenizer
import os
import re
from torchvision import transforms

def train_hrgr():
    # 1. Load Config
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Resources (Templates & Vocab)
    with open("data/processed/templates.json", "r", encoding="utf-8") as f:
        templates = json.load(f)
    vocab = WordVocabulary.load("data/processed/vocab.json")
    
    # 3. Data Loading
    import pandas as pd
    data_path = config['data'].get('train_csv', "data/processed/iu_xray_dataset_raw.csv")
    print(f"📂 Loading data from: {data_path}")
    train_df = pd.read_csv(data_path)
    
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ở đây chúng ta cần dataset trả về raw_report
    dataset = MedicalImageTextDataset(
        train_df, 
        image_transform=transform, 
        text_tokenizer=AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 4. Model Initialization
    model = HRGRAgent(
        image_encoder_name=config['model']['image_encoder'],
        vocab_size=len(vocab),
        templates=templates
    )
    
    # 4.5 Load Pre-trained Image Encoder Weights (Robust Version)
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Đang nạp trọng số từ: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Xử lý cả 2 trường hợp: có model_state_dict hoặc không
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Lọc và sửa tên layers cho khớp với kiến trúc SwinV2 của timm 
        encoder_weights = {}
        for k, v in state_dict.items():
            if 'image_encoder.model.' in k:
                new_k = k.split('image_encoder.model.')[-1]
                # Sửa format layers (ví dụ: layers.0. thành layers_0.)
                for i in range(4):
                    new_k = new_k.replace(f'layers.{i}.', f'layers_{i}.')
                encoder_weights[new_k] = v
        
        if encoder_weights:
            msg = model.image_encoder.model.load_state_dict(encoder_weights, strict=False)
            print(f"✅ Nạp thành công {len(encoder_weights)} layers. Status: {msg}")
        else:
            print("⚠️ Cảnh báo: Không tìm thấy trọng số tương thích với Image Encoder.")
    
    # 5. Đóng băng Image Encoder (Freeze) - Cần thiết cho bản Base hội tụ ổn định
    print("❄️ Freezing Image Encoder for stable training...")
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    # 5. Khởi tạo Trainer
    trainer = HRGRRLTrainer(model, vocab, templates, config, device=device)

    # 5.1 Logic Tự động Resume từ checkpoint mới nhất
    start_epoch = 1
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tìm file checkpoint có số epoch lớn nhất: hrgr_epoch_X.pth
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "hrgr_epoch_*.pth"))
    if checkpoints:
        # Lấy số epoch từ tên file
        epochs_found = [int(re.findall(r'epoch_(\d+)', f)[0]) for f in checkpoints]
        if epochs_found:
            latest_epoch = max(epochs_found)
            latest_ckpt = os.path.join(checkpoint_dir, f"hrgr_epoch_{latest_epoch}.pth")
            print(f"🔄 Tìm thấy checkpoint epoch {latest_epoch}. Đang nạp để chạy tiếp...")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device), strict=False)
            start_epoch = latest_epoch + 1
            # Cập nhật scheduler tương ứng
            for _ in range(latest_epoch):
                trainer.scheduler.step()
            print(f"⏩ Sẽ bắt đầu huấn luyện từ Epoch {start_epoch}")
    
    # 6. Training Loop (MLE Phase)
    print(f"🚀 Starting MLE Training from Epoch {start_epoch} to {config['training']['epochs']}...")
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # TỰ ĐỘNG UNFREEZE SAU EPOCH 5 ĐỂ TỐI ƯU R@1
        if epoch == 6:
            trainer.unfreeze_encoder(encoder_lr=2e-6)
        elif start_epoch > 5 and epoch == start_epoch: # Nếu resume từ epoch 6 trở đi
            trainer.unfreeze_encoder(encoder_lr=2e-6)

        loss = trainer.train_epoch_mle(dataloader, epoch)
        # Cập nhật LR
        trainer.scheduler.step()
        print(f"Epoch {epoch} Loss: {loss:.4f} | LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpiont (Local và Backup vào Drive nếu đang chạy Colab)
        local_path = f"checkpoints/hrgr_epoch_{epoch}.pth"
        torch.save(model.state_dict(), local_path)
        
        # --- TỰ ĐỘNG BACKUP VÀO BÊN TRONG GOOGLE DRIVE ---
        drive_dir = "/content/drive/MyDrive/HRGR_Checkpoints"
        if os.path.exists("/content/drive/MyDrive"): # Kiểm tra xem có đang nối Drive không
            import shutil
            os.makedirs(drive_dir, exist_ok=True)
            drive_path = os.path.join(drive_dir, f"hrgr_epoch_{epoch}.pth")
            shutil.copy(local_path, drive_path)
            print(f"✅ Đã chép an toàn Epoch {epoch} vào Google Drive: {drive_path}")
        # ---------------------------------------------------

if __name__ == "__main__":
    train_hrgr()
