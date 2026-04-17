import torch
import torch.optim as optim
import copy
from tqdm import tqdm
from src.losses.contrastive import ClusteringGuidedContrastiveLoss

class MultimodalTrainer:
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # (Đã loại bỏ EMA module nhằm tiết kiệm thẻ nhớ VRAM hỗ trợ Batch lớn)

        # Thêm GradScaler để dùng Mixed Precision (FP16) tăng tốc x2
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
        
        # --- TỐI ƯU: Discriminative Learning Rate (Tốc độ học phân biệt) ---
        # Ngăn việc training quá mạnh làm hỏng kiến thức của Pre-trained backbones
        lr_projector = float(config['training']['lr'])
        lr_backbone = lr_projector / 20.0 # Theo benchmark ảnh tham khảo
        
        # Xử lý tương thích với lớp bọc nn.DataParallel 
        model_module = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        backbone_params = list(model_module.image_encoder.parameters()) + list(model_module.text_encoder.parameters())
        projector_params = list(model_module.image_proj.parameters()) + list(model_module.text_proj.parameters())
        
        params_groups = [
            {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
            {'params': projector_params, 'lr': lr_projector, 'name': 'projector'}
        ]
        
        self.optimizer = optim.AdamW(
            params_groups, 
            weight_decay=float(config['training']['weight_decay'])
        )
        
        self.criterion = ClusteringGuidedContrastiveLoss(
            temperature=config['model']['temperature']
        ).to(device)

        # --- TỐI ƯU: Scheduler với Warmup ---
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        epochs = config['training']['epochs']

        # 1. Giai đoạn Warmup: Tăng dần LR từ 1/100 -> full LR
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.1,  # <--- Khởi đầu từ 10% LR thay vì 1% để tránh bị kẹt
            total_iters=warmup_epochs
        )
        
        # 2. Giai đoạn Hội tụ: Giảm dần theo hình Cos
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-6
        )

        # Kết hợp thành chuỗi: Warmup xong rồi mới tới Cosine
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )


    def get_lr(self):
        """Trả về tuple (Backbone LR, Projector LR)"""
        return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']
        
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
                if (i + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    # max_norm=5.0 phù hợp với SwinV2 (grad norm thường ~3-8)
                    # Trước đây 1.0 → cắt gradient quá mạnh → model không học được
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            lr_b, lr_p = self.get_lr()
            total_loss += loss.item() * accum_steps
            pbar.set_postfix({
                "Loss": f"{loss.item() * accum_steps:.4f}",
                "LR_b": f"{lr_b:.2e}",
                "LR_p": f"{lr_p:.2e}"
            })
            
        avg_loss = total_loss / (max_steps if max_steps else len(dataloader))
        # Cập nhật Scheduler vào cuối vòng lặp Epoch
        self.scheduler.step()
        lr_b, lr_p = self.get_lr()
        print(f"✅ Hết Epoch {epoch} - Trung bình Loss: {avg_loss:.4f} | LR_b: {lr_b:.2e} | LR_p: {lr_p:.2e}")
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
