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
from torchvision import transforms

def train_hrgr():
    # 1. Load Config
    with open("configs/default.yaml", "r") as f:
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
    
    # 5. Trainer
    trainer = HRGRRLTrainer(model, vocab, templates, config, device=device)

    # 6. Training Loop (MLE Phase)
    print("Starting MLE Training...")
    for epoch in range(1, 6): # Huấn luyện 5 epoch MLE mẫu
        loss = trainer.train_epoch_mle(dataloader, epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
        
        # Save checkpiont
        torch.save(model.state_dict(), f"checkpoints/hrgr_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_hrgr()
