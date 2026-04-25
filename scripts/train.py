import os
import yaml
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

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

def get_train_transforms(image_size):
    """Data Augmentation cho tập Train để chống Overfitting"""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)), # Phóng to để chuẩn bị cắt
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)), # Cắt mạnh hơn một chút
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), # Biến dạng hình học
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Tạo góc nhìn giả lập
        transforms.RandomHorizontalFlip(), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # Chỉnh màu mạnh hơn
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # Làm mờ ngẫu nhiên để mô hình không bám vào nhiễu pixel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size):
    """Transform chuẩn cho tập Validation/Test"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    print("==================================================================")
    print("🚀 LUẬN VĂN: HUẤN LUYỆN MULTIMODAL SOFT CLUSTERING-GUIDED NEGATIVE SAMPLING 🚀")
    print("==================================================================")
    
    # Kích hoạt sử dụng GPU để train Nhanh hơn (Nếu máy bạn có Card rời NVIDIA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 AI đang tính toán bằng lõi: {device.upper()}")
    
    config = load_config()
    
    # 1. GIAI ĐOẠN DATASET
    print("\n[1] Đang nạp bộ từ điển Tokenizer của mô hình ClinicalBERT...")
    # Tải Tokenizer chuyên dụng cho dữ liệu sinh khoa/y tế
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    train_transform = get_train_transforms(config['data']['image_size'])
    val_transform = get_val_transforms(config['data']['image_size'])
    
    print("[2] Đang Load Database...")
    try:
        train_df = pd.read_csv(config['data']['train_csv'])
        val_df_old = pd.read_csv(config['data']['val_csv'])
        
        # --- MỚI: Load nhãn cụm MỀM (Soft Labels) ---
        processed_dir = "data/processed"
        train_soft = np.load(os.path.join(processed_dir, "soft_labels_train.npy"))
        val_soft_old = np.load(os.path.join(processed_dir, "soft_labels_val.npy"))
        
        # --- CHIA TẬP VAL CŨ THÀNH VAL MỚI VÀ EVAL ---
        # Tách 50% tập Val cũ làm tập Eval chuyên dụng cho đánh giá R@1
        # Điều này giúp theo dõi Val Loss (trên Val mới) và Metric (trên Eval) độc lập
        val_df, eval_df, val_soft, eval_soft = train_test_split(
            val_df_old, val_soft_old, test_size=0.5, random_state=42
        )
        print(f"✅ Đã chia tập Val cũ ({len(val_df_old)}) -> Val mới: {len(val_df)} | Eval: {len(eval_df)}")
        print(f"✅ Đã nạp nhãn mềm GMM: Train {train_soft.shape} | Val {val_soft.shape} | Eval {eval_soft.shape}")
        
        # --- FULL DATASET: Gộp tất cả để đánh giá R@K nếu cần ---
        full_df   = pd.concat([train_df, val_df, eval_df], ignore_index=True)
        full_soft = np.concatenate([train_soft, val_soft, eval_soft], axis=0)
        print(f"✅ Full Dataset: {len(full_df)} mẫu | Soft Labels {full_soft.shape}")
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
        print("💡 Chú ý: Hãy phải chạy Python trên 2 file prepare_dataset.py và create_clusters.py trước!")
        return

    train_dataset = MedicalImageTextDataset(train_df, train_transform, tokenizer, config['data']['max_text_length'], soft_labels=train_soft)
    val_dataset   = MedicalImageTextDataset(val_df,   val_transform,   tokenizer, config['data']['max_text_length'], soft_labels=val_soft)
    eval_dataset  = MedicalImageTextDataset(eval_df,  val_transform,   tokenizer, config['data']['max_text_length'], soft_labels=eval_soft)
    
    # Full dataset — không augmentation, dùng cho evaluate_retrieval nếu cần
    full_dataset  = MedicalImageTextDataset(full_df,  val_transform,   tokenizer, config['data']['max_text_length'], soft_labels=full_soft)
    
    # Sửa lỗi Threading/Crash của hệ điều hành Windows khi gọi num_workers > 0
    num_workers = 0 if os.name == 'nt' else config['training']['num_workers']
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
    eval_loader  = DataLoader(eval_dataset,  batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
    # Full loader: đánh giá R@K trên toàn bộ dataset
    full_loader  = DataLoader(full_dataset,  batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
    
    # 2. GIAI ĐOẠN KHỞI TẠO NEURAL NETWORK
    print("\n[3] 🏗️ Đang xây dựng cấu trúc Mạng Neuron (Swin Transformer V2 + ClinicalBERT + MLP)...")
    model = MultimodalModel(
        image_encoder_name=config['model']['image_encoder'],
        text_model_name=config['model']['text_encoder'],
        embed_dim=config['model']['embed_dim']
    )
    
    print(f"Tổng số tham số mạng lưu giữ: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} Parameters")
    
    # --- ÁP DỤNG MULTI-GPU NẾU CÓ ---
    if torch.cuda.device_count() > 1:
        print(f"🔥 Kích hoạt tự động {torch.cuda.device_count()} GPUs chạy song song (DataParallel) 🔥")
        model = torch.nn.DataParallel(model)

    # 3. GIAI ĐOẠN TIẾN HÀNH TRAINING LOOP
    print("\n[4] ⚡ TIẾN HÀNH ĐÀO TẠO MÔ HÌNH VỚI DỮ LIỆU...")
    trainer = MultimodalTrainer(model, config, device=device)
    
    epochs = config['training']['epochs']
    best_r1 = -1.0

    # --- TRACKING XU HUONG R@1 (Rolling Average 5 epoch) ---
    # Muc dich: Loai bo nhieu epoch-by-epoch, nhin thay xu huong thuc su
    # Tai muc R@1=2-3%, moi epoch dao dong chi la 1-2 mau -> nhieu nguyen chat
    r1_history = []
    patience   = 0
    # ---------------------------------------------------

    # Tao thu muc chua tep file .pth Weights cuoi cung
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, "training_history.csv")
    history = []
    
    # --- [RESUME LOGIC] ---
    start_epoch = 1
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if os.path.exists(last_ckpt_path):
        checkpoint = trainer.load_checkpoint(last_ckpt_path)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_r1 = checkpoint.get('best_r1', -1.0)
            print(f"⏩ Đang tiếp tục huấn luyện từ Epoch {start_epoch}...")
            # Load history if exists
            if os.path.exists(history_path):
                try:
                    history = pd.read_csv(history_path).to_dict('records')
                except:
                    pass # history remains []
    # ----------------------

    eval_every = config['training'].get('eval_every_n_epochs', 1)

    for epoch in range(start_epoch, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)
        
        # --- ĐÁNH GIÁ R@1 (Theo định kỳ) ---
        current_r1 = 0.0
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"\n📊 Đang đánh giá R@1 (Clustering-Guided) cho Epoch {epoch}...")
            try:
                # [ĐÃ SỬA] Đánh giá trên tập EVAL chuyên biệt (tách từ Val cũ) để khách quan hơn
                r_strict, r_cluster = evaluate_retrieval(trainer.model, eval_loader, device)
                
                print(f"✅ Epoch {epoch} [Strict]  - R@1: {r_strict[0]:.2f}% | R@5: {r_strict[1]:.2f}%")
                print(f"✅ Epoch {epoch} [Cluster] - R@1: {r_cluster[0]:.2f}% | R@5: {r_cluster[1]:.2f}% | R@10: {r_cluster[2]:.2f}%")
                
                # --- LƯU NHẬT KÝ (LOGGING) ---
                history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'r1_strict': r_strict[0],
                    'r5_strict': r_strict[1],
                    'r1_cluster': r_cluster[0],
                    'r5_cluster': r_cluster[1],
                    'r10_cluster': r_cluster[2]
                })
                pd.DataFrame(history).to_csv(history_path, index=False)
                
                current_r1 = r_cluster[0]
            except Exception as e:
                print(f"⚠️ Lỗi khi đánh giá R@1: {e}")
        else:
            print(f"⏩ Epoch {epoch}: Bỏ qua đánh giá R@K (Config: eval_every_n_epochs={eval_every})")
            # Lưu log cơ bản
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            pd.DataFrame(history).to_csv(history_path, index=False)
            
        # Lưu checkpoint tốt nhất (chỉ lưu weights để suy luận/inference)
        is_best = False
        if current_r1 > best_r1 and current_r1 > 0:
            best_r1 = current_r1
            is_best = True
            ckpt_path = os.path.join(config['training']['checkpoint_dir'], "best_model.pth")
            
            # Xuất đúng định dạng Weight dù có dùng DataParallel hay không
            model_module = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
            torch.save(model_module.state_dict(), ckpt_path)
            print(f"⭐ [CÓ CẢI THIỆN R@1 = {best_r1:.2f}%] - Lưu weights vào: {ckpt_path}")

        # --- LUÔN LƯU FULL CHECKPOINT ĐỂ RESUME ---
        trainer.save_checkpoint(last_ckpt_path, epoch, best_r1)
        
        # Lưu bản weigths cuối cùng (last_model.pth) để dễ so sánh
        last_weights_path = os.path.join(config['training']['checkpoint_dir'], "last_model.pth")
        model_module = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
        torch.save(model_module.state_dict(), last_weights_path)

        # --- TỰ ĐỘNG BACKUP VÀO BÊN TRONG GOOGLE DRIVE (CHO COLAB) ---
        drive_roots = ["/content/drive/MyDrive", "/content/drive/My Drive"]
        for root in drive_roots:
            if os.path.exists(root):
                drive_dir = os.path.join(root, "Multimodal_Checkpoints")
                os.makedirs(drive_dir, exist_ok=True)
                
                # Backup bản best
                if is_best:
                    shutil.copy(ckpt_path, os.path.join(drive_dir, "best_model.pth"))
                    print(f"✅ Đã backup best_model.pth vào Drive: {drive_dir}")
                
                # Backup bản full checkpoint để đề phòng mất session Colab
                shutil.copy(last_ckpt_path, os.path.join(drive_dir, "last_checkpoint.pth"))
                print(f"✅ Đã backup last_checkpoint.pth vào Drive")
                break

        # --- LƯU Ý CHO KAGGLE ---
        if os.path.exists("/kaggle/working"):
            # Trên Kaggle, mọi thứ trong /kaggle/working sẽ được lưu khi dùng "Save Version"
            # Ta đã lưu vào checkpoints/ nên không cần copy đi đâu thêm.
            pass

        # --- ROLLING AVERAGE: Xu huong R@1 thuc su ---
        # Tai muc 2-3%, thay doi 0.07% = 1 mau flip -> nhieu thuan tuy
        # Rolling avg 5 epoch cho thay TREND thuc su cua qua trinh hoc
        r1_history.append(current_r1)
        W = 5
        if len(r1_history) >= W:
            avg_now  = sum(r1_history[-W:]) / W
            avg_prev = sum(r1_history[-W-1:-1]) / W if len(r1_history) > W else avg_now
            if avg_now > avg_prev + 0.01:
                arrow = "TANG THAT SU"
                patience = 0
            elif avg_now < avg_prev - 0.01:
                arrow = "GIAM THAT SU"
                patience += 1
            else:
                arrow = "ON DINH"
                patience += 1
            print(f"   Rolling Avg [{W}-ep]: {avg_now:.2f}%  [{arrow}]  Best: {best_r1:.2f}%")
            if patience >= 15:
                print("   WARNING: 15 epoch khong cai thien. Nen kiem tra loss co dang giam khong.")
        # -----------------------------------------------

    print("\n HOAN TAT QUA TRINH THUC THI KHOA LUAN!")

if __name__ == "__main__":
    main()
