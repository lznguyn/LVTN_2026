import torch
import torch.optim as optim
from tqdm import tqdm
from src.losses.contrastive import ClusteringGuidedContrastiveLoss

class MultimodalTrainer:
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Khởi tạo thuật toán Toán học tối ưu (Optimizer) để cập nhật độ dốc
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(config['training']['lr']), 
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # Hàm lõi để bỏ đi âm tính giả (Nhân tố then chốt cho đề tài của bạn)
        self.criterion = ClusteringGuidedContrastiveLoss(
            temperature=config['model']['temperature']
        ).to(device)
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
        for batch in pbar:
            # 1. Đưa dữ liệu qua Card Đồ họa (GPU) để tăng tốc nếu có
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            cluster_ids = batch['cluster_id'].to(self.device, non_blocking=True)
            
            # Xóa các độ lớn gradient thặng dư của batch trước
            self.optimizer.zero_grad()
            
            # 2. Truyền xuôi Model (Forward Pass)
            img_embeds, txt_embeds = self.model(
                images, input_ids, attention_mask
            )
            
            # 3. Tính hàm mất mát
            loss = self.criterion(img_embeds, txt_embeds, cluster_ids)
            
            # 4. Lan truyền ngược (Backward Pass)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Hết Epoch {epoch} - Trung bình Loss huấn luyện: {avg_loss:.4f}")
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
