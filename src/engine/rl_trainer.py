import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
import re

class HRGRRLTrainer:
    def __init__(self, model, vocab, templates, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.templates = templates
        self.device = device
        self.config = config
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # 0. Tính toán trọng số Class Weights để phá vỡ con số 62.11%
        self.weights = self._calculate_class_weights().to(device)
        print(f"⚖️ Applied Class Weights for Policy: {self.weights[:5].tolist()}...")

        self.criterion_policy = nn.CrossEntropyLoss(weight=self.weights)
        self.criterion_stop = nn.BCEWithLogitsLoss()
        self.criterion_word = nn.CrossEntropyLoss(ignore_index=0) # Index 0 is <pad>
        
        # 0. Thêm Schedulers (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs'], 
            eta_min=1e-6
        )
        
        # 0.1 Thêm GradScaler cho Mixed Precision (FP16)
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    def unfreeze_encoder(self, encoder_lr=2e-6):
        """
        Mở khóa Image Encoder để bắt đầu Fine-tuning sâu hơn.
        Sử dụng LR nhỏ hơn cho Encoder để tránh làm hỏng trọng số đã pre-trained.
        """
        print(f"🔥 UNFREEZING Image Encoder with LR: {encoder_lr}")
        for param in self.model.image_encoder.parameters():
            param.requires_grad = True
            
        # Cập nhật Optimizer để bao gồm cả tham số encoder với LR khác nhau
        self.optimizer = optim.AdamW([
            {'params': self.model.image_encoder.parameters(), 'lr': encoder_lr},
            {'params': [p for n, p in self.model.named_parameters() if 'image_encoder' not in n], 'lr': float(self.config['training']['lr'])}
        ], weight_decay=float(self.config['training']['weight_decay']))
        
        # Reset scheduler để tương thích với optimizer mới
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['training']['epochs'], 
            eta_min=1e-6
        )

    def _calculate_class_weights(self):
        """
        Tính toán trọng số nghịch đảo dựa trên tần suất của các Action trong tập Train.
        Ép mô hình phải học các Template thay vì chỉ đoán 'Generate' (62%).
        """
        import pandas as pd
        from collections import Counter
        
        train_csv = self.config['data'].get('train_csv', "data/splits/train.csv")
        if not os.path.exists(train_csv):
            return torch.ones(len(self.templates) + 1)
            
        df = pd.read_csv(train_csv)
        all_actions = []
        for report in df['report'].dropna():
            # Sử dụng logic tách câu tương tự get_targets
            sentences = re.split(r'[.;?!\n]', str(report))
            for s in sentences:
                s = s.strip()
                if not s: continue
                
                action = 0
                for idx, t in enumerate(self.templates):
                    if s.lower() == t.lower():
                        action = idx + 1
                        break
                all_actions.append(action)
                break # Chỉ tính action của câu đầu tiên để tối ưu R@1
        
        counts = Counter(all_actions)
        num_classes = len(self.templates) + 1
        weights = torch.ones(num_classes)
        
        total = sum(counts.values())
        for i in range(num_classes):
            count = counts.get(i, 1) # Tránh chia cho 0
            # Công thức: weight = Total / (n_classes * count)
            # Dùng mũ 0.8 để trọng số mạnh hơn mũ 0.5 cũ, ép model học lớp hiếm
            weights[i] = (total / (count)) ** 0.8 
            
        # Chuẩn hóa để trọng số trung bình = 1
        weights = weights / weights.mean()
        return weights

    def get_targets(self, report):
        """
        Tách báo cáo thành các mục tiêu: {Action, Stop, Words} cho từng câu.
        """
        sentences = re.split(r'[.;?!\n]', str(report))
        targets = []
        for i, s in enumerate(sentences):
            s = s.strip()
            if not s: continue
            
            # Kiểm tra xem có khớp template nào không?
            action = 0
            for idx, t in enumerate(self.templates):
                if s.lower() == t.lower():
                    action = idx + 1
                    break
            
            # Nếu không khớp, lấy tokens cho Word Generator
            tokens = []
            if action == 0:
                tokens = [self.vocab(w) for w in re.findall(r'\w+', s.lower())]
            
            # Stop = 1 nếu là câu cuối, ngược lại = 0
            stop = 1.0 if i == len(sentences) - 1 else 0.0
            
            targets.append({
                'action': action,
                'stop': stop,
                'tokens': tokens
            })
            
            if i >= 7: # Tối đa 8 câu theo config
                break
        return targets

    def prepare_batch_targets(self, reports, max_sentences=8, max_words=20):
        """
        Chuẩn bị Tensor targets cho cả Batch.
        """
        batch_size = len(reports)
        target_actions = torch.zeros((batch_size, max_sentences), dtype=torch.long)
        target_stops = torch.zeros((batch_size, max_sentences), dtype=torch.float)
        target_words = torch.zeros((batch_size, max_sentences, max_words), dtype=torch.long)
        
        for i, report in enumerate(reports):
            img_targets = self.get_targets(report)
            for j, t in enumerate(img_targets):
                if j >= max_sentences: break
                target_actions[i, j] = t['action']
                target_stops[i, j] = t['stop']
                
                # Copy tokens vào target_words
                tokens = t['tokens'][:max_words]
                for k, token_id in enumerate(tokens):
                    target_words[i, j, k] = token_id
                    
        return target_actions.to(self.device), target_stops.to(self.device), target_words.to(self.device)

    def train_epoch_mle(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        # 0. Lấy max_steps từ cấu hình
        max_steps = self.config['training'].get('max_steps_per_epoch', None)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} MLE Training")
        
        for i, batch in enumerate(pbar):
            if max_steps and i >= max_steps:
                print(f"Reached max steps: {max_steps}. Finishing epoch early.")
                break
                
            images = batch['image'].to(self.device)
            old_images = batch['image_old'].to(self.device)
            reports = batch['raw_report']
            
            # 1. Prepare Targets
            t_actions, t_stops, t_words = self.prepare_batch_targets(reports)
            
            # 2. Forward với chế độ AMP Autocast
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                p_logits, s_logits, w_logits = self.model(images, old_images, t_actions, t_words)
                
                # 3. Calculate Losses
                loss_p = self.criterion_policy(p_logits.reshape(-1, p_logits.size(-1)), t_actions.reshape(-1))
                loss_s = self.criterion_stop(s_logits.squeeze(-1), t_stops)
                loss_w = self.criterion_word(w_logits.reshape(-1, w_logits.size(-1)), t_words.reshape(-1))
                
                # Nhân 5.0 cho Policy Loss để ép mô hình ưu tiên chọn đúng Template/Bệnh lý
                loss = 5.0 * loss_p + loss_s + loss_w
            
            # 4. Backward & Step với Scaler
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        return total_loss / (i + 1)

    def calculate_reward(self, gen_reports, gt_reports):
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        rewards = []
        for gen, gt in zip(gen_reports, gt_reports):
            gen_tokens = re.findall(r'\w+', gen.lower())
            gt_tokens = re.findall(r'\w+', gt.lower())
            score = sentence_bleu([gt_tokens], gen_tokens, smoothing_function=smooth)
            rewards.append(score)
        return torch.tensor(rewards, device=self.device)

    def train_epoch_rl(self, dataloader, epoch):
        """
        Skeleton cho giai đoạn RL. 
        Giai đoạn này yêu cầu mode Sampling từ model.
        """
        self.model.train()
        print("Giai đoạn RL yêu cầu triển khai Sampling logic trong model (REINFORCE).")
        return 0.0
