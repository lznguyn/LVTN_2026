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

        # ── EMA Model (Exponential Moving Average) ──────────────────────────
        # Mục đích: Giữ bản trung bình trượt của weights qua các bước train.
        # Khi đánh giá R@1, dùng ema_model thay vì model thô đang train.
        # EMA loại bỏ nhiễu gradient ngắn hạn → R@1 tăng đều, không giật.
        # Công thức: ema_w = tau * ema_w + (1-tau) * current_w
        # tau=0.999 → mỗi bước EMA giữ 99.9% giá trị cũ, nhích 0.1% về mới.
        self.ema_model = copy.deepcopy(model).to(device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)  # EMA không tham gia gradient
        self.ema_tau = config['training'].get('ema_tau', 0.999)
        # ────────────────────────────────────────────────────────────────────
        
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

        # Warmup 5 epoch đầu: LR tăng dần từ 0 -> lr
        # Sau đó Cosine Annealing giảm dần đến eta_min
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        total_epochs = config['training']['epochs']
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,  # Bắt đầu từ 1% LR
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

    def _update_ema(self):
        """Cập nhật EMA weights sau mỗi optimizer step."""
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_tau).add_(model_p.data, alpha=1.0 - self.ema_tau)

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
                if (i + 1) % accum_steps == 0:
                    # Gradient Clipping: Tránh gradient spike làm R@1 giật lùi
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self._update_ema()  # ← Cập nhật EMA sau mỗi real optimizer step
            else:
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_ema()  # ← Cập nhật EMA sau mỗi real optimizer step
            
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
