import torch
import torch.optim as optim
from tqdm import tqdm
from src.losses.contrastive import ClusteringGuidedContrastiveLoss

class MultimodalTrainer:
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Thêm GradScaler để dùng Mixed Precision (FP16) tăng tốc x2
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
        
        # Khởi tạo thuật toán Toán học tối ưu (Optimizer)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(config['training']['lr']), 
            weight_decay=float(config['training']['weight_decay'])
        )
        
        self.criterion = ClusteringGuidedContrastiveLoss(
            temperature=config['model']['temperature']
        ).to(device)
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        # Giới hạn số bước nếu config có yêu cầu (để chạy 50 epoch nhanh)
        max_steps = self.config['training'].get('max_steps_per_epoch', None)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
        for i, batch in enumerate(pbar):
            if max_steps and i >= max_steps:
                break
                
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            cluster_ids = batch['cluster_id'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # --- TỐI ƯU CỰC MẠNH: Dùng Autocast cho FP16 ---
            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                img_embeds, txt_embeds = self.model(images, input_ids, attention_mask)
                loss = self.criterion(img_embeds, txt_embeds, cluster_ids)
            
            # Backward với Scaler để tránh lỗi tràn số khi dùng FP16
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / (max_steps if max_steps else len(dataloader))
        print(f"✅ Hết Epoch {epoch} - Trung bình Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        
        # Không tính và lưu gradient trong quá trình Test model
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} Validation")
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                cluster_ids = batch['cluster_id'].to(self.device, non_blocking=True)
                
                img_embeds, txt_embeds = self.model(
                    images, input_ids, attention_mask
                )
                
                loss = self.criterion(img_embeds, txt_embeds, cluster_ids)
                total_loss += loss.item()
                pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
                
        avg_loss = total_loss / len(dataloader)
        print(f"📊 Hết Epoch {epoch} - Trung bình Loss xác thực (Val): {avg_loss:.4f}")
        return avg_loss
