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
        
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_stop = nn.BCEWithLogitsLoss()
        self.criterion_word = nn.CrossEntropyLoss(ignore_index=0) # Index 0 is <pad>

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
        
        # 0. AMP Scalar cho huấn luyện FP16 tăng tốc độ
        scaler = torch.amp.GradScaler('cuda')
        
        max_steps = self.config['training'].get('max_steps_per_epoch', None)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} MLE Training")
        
        for i, batch in enumerate(pbar):
            if max_steps and i >= max_steps:
                print(f"Reached max steps: {max_steps}. Finishing epoch early.")
                break
                
            images = batch['image'].to(self.device)
            reports = batch['raw_report']
            
            # 1. Prepare Targets
            t_actions, t_stops, t_words = self.prepare_batch_targets(reports)
            
            # 2. Forward với chế độ AMP Autocast
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                p_logits, s_logits, w_logits = self.model(images, t_actions, t_words)
                
                # 3. Calculate Losses
                loss_p = self.criterion_policy(p_logits.reshape(-1, p_logits.size(-1)), t_actions.reshape(-1))
                loss_s = self.criterion_stop(s_logits.squeeze(-1), t_stops)
                loss_w = self.criterion_word(w_logits.reshape(-1, w_logits.size(-1)), t_words.reshape(-1))
                
                loss = loss_p + loss_s + loss_w
            
            # 4. Backward & Step với Scaler
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
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
