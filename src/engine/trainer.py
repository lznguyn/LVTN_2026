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

        # Thêm Scheduler: Giảm tốc độ học theo đồ thị hình sin (Cosine Annealing)
        # Giúp mô hình hội tụ ổn định hơn
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs'], 
            eta_min=1e-6
        )
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        max_steps = self.config['training'].get('max_steps_per_epoch', None)
        # Lấy số bước tích lũy từ config (mặc định là 1 nếu không có)
        accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
        for i, batch in enumerate(pbar):
            if max_steps and i >= max_steps:
                break
                
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            cluster_ids = batch['cluster_id'].to(self.device, non_blocking=True)
            
            # --- TỐI ƯU: Mixed Precision + Accumulation ---
            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                img_embeds, txt_embeds = self.model(images, input_ids, attention_mask)
                loss = self.criterion(img_embeds, txt_embeds, cluster_ids)
                # Chia loss cho số bước tích lũy
                loss = loss / accum_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                # Chỉ cập nhật trọng số sau mỗi accum_steps
                if (i + 1) % accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * accum_steps
            pbar.set_postfix({
                "Loss": f"{loss.item() * accum_steps:.4f}",
                "LR": f"{self.get_lr():.2e}"
            })
            
        avg_loss = total_loss / (max_steps if max_steps else len(dataloader))
        # Cập nhật Scheduler vào cuối vòng lặp Epoch
        self.scheduler.step()
        print(f"✅ Hết Epoch {epoch} - Trung bình Loss: {avg_loss:.4f} | LR mới: {self.get_lr():.2e}")
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
