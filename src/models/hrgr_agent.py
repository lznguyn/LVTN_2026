import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import SwinTransformerV2Encoder
import json

class Attention(nn.Module):
    """
    Bahdanau Attention mechanism for spatial features.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (Batch, NumPixels, EncoderDim)
        # decoder_hidden: (Batch, DecoderDim)
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))) # (Batch, NumPixels, 1)
        alpha = self.softmax(att.squeeze(2)) # (Batch, NumPixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (Batch, EncoderDim)
        return attention_weighted_encoding, alpha

class HRGRAgent(nn.Module):
    def __init__(self, image_encoder_name, vocab_size, templates, embed_dim=512, decoder_dim=512):
        super(HRGRAgent, self).__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.templates = templates # Danh sách các câu mẫu
        self.num_templates = len(templates)

        # Image Encoder - Lấy cả global và spatial features
        self.image_encoder = SwinTransformerV2Encoder(model_name=image_encoder_name, features_only=True)
        self.encoder_dim = self.image_encoder.feature_dim
        
        # Sentence Decoder (Hierarchical level)
        self.sentence_decoder = nn.GRUCell(self.encoder_dim, decoder_dim)
        
        # Policy Head (Chọn: Tự viết [0] hoặc Lấy mẫu [1..N])
        self.policy_head = nn.Linear(decoder_dim, 1 + self.num_templates)
        
        # Stop Control Head (0: Tiếp tục, 1: Dừng)
        self.stop_head = nn.Linear(decoder_dim, 1)

        # Word Generator (Word-level)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(self.encoder_dim, decoder_dim, embed_dim)
        self.word_decoder = nn.GRUCell(embed_dim + self.encoder_dim, decoder_dim)
        self.word_fc = nn.Linear(decoder_dim, vocab_size)
        
        # Khởi tạo hidden state từ Image Global Features (Hỗ trợ Dual Image Fusion)
        self.init_h = nn.Linear(self.encoder_dim * 2, decoder_dim) # Fusion of 2 images
        self.global_fusion = nn.Linear(self.encoder_dim * 2, self.encoder_dim) # To feed into GRU

    def get_global_features(self, spatial_features):
        """
        spatial_features: (B, H, W, C) cho SwinV2 (timm features_only)
        """
        if spatial_features.dim() == 4:
            # (B, H, W, C) -> mean over H, W -> (B, C)
            global_features = spatial_features.mean(dim=[1, 2])
        else:
            # (B, C, H, W) -> (B, C)
            global_features = spatial_features.mean(dim=[2, 3])
        return global_features

    def init_hidden_state(self, global_features):
        h = self.init_h(global_features)
        return torch.tanh(h)

    def forward(self, images, old_images=None, target_actions=None, target_words=None, max_sentences=8, max_words=20):
        """
        Forward pass hỗ trợ cả Training (MLE) và Inference.
        Chấp nhận thêm old_images để thực hiện báo cáo so sánh.
        """
        batch_size = images.size(0)
        device = images.device

        # Nếu không có ảnh cũ, dùng chính ảnh mới làm baseline
        if old_images is None:
            old_images = images

        # 1. Image features (Shared Encoder)
        spatial_new = self.image_encoder(images) # (B, H, W, C)
        global_new = self.get_global_features(spatial_new)
        
        spatial_old = self.image_encoder(old_images)
        global_old = self.get_global_features(spatial_old)

        # Fusion Global Features
        global_combined = torch.cat([global_new, global_old], dim=1) # (B, 2*C)
        global_fused = self.global_fusion(global_combined) # (B, C) - used for cell update
        
        # Merge Spatial Features (Concatenate samples)
        # Flatten spatial: (B, NumPixels, C)
        if spatial_new.dim() == 4:
             spatial_new = spatial_new.reshape(batch_size, -1, self.encoder_dim)
             spatial_old = spatial_old.reshape(batch_size, -1, self.encoder_dim)
        else:
             spatial_new = spatial_new.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, self.encoder_dim)
             spatial_old = spatial_old.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, self.encoder_dim)
        
        # Concatenate spatial along sequence dim: (B, 2*NumPixels, C)
        spatial_features = torch.cat([spatial_new, spatial_old], dim=1)
        
        # 2. Init Sentence Decoder (From fused global)
        h_s = self.init_hidden_state(global_combined)
        
        all_policy_logits = []
        all_stop_logits = []
        all_word_logits = [] # (B, max_sentences, max_words, vocab_size)

        for i in range(max_sentences):
            # Cập nhật topic vector q_i (h_s) sử dụng global_fused
            h_s = self.sentence_decoder(global_fused, h_s)
            
            # Policy & Stop Preds
            policy_logits = self.policy_head(h_s)
            stop_logits = self.stop_head(h_s)
            
            all_policy_logits.append(policy_logits)
            all_stop_logits.append(stop_logits)
            
            # --- Word Decoding (Sinh câu mới) ---
            # Chỉ thực hiện nếu tại bước này action đang là 0 (Generate)
            # Trong training MLE, chúng ta tính loss Word cho tất cả các câu mà target_action == 0
            
            h_w = h_s # Init hidden word decoder từ topic state
            sentence_word_logits = []
            
            # Input đầu tiên cho Word Decoder là <start> token
            prev_word = torch.full((batch_size,), 1, dtype=torch.long, device=device) # <start> = 1
            
            for t in range(max_words):
                embeddings = self.word_embedding(prev_word) # (B, embed_dim)
                # Attention over spatial_features
                attention_weighted_encoding, alpha = self.attention(spatial_features, h_w)
                
                # GRU Step
                word_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
                h_w = self.word_decoder(word_input, h_w)
                
                # Predict word
                word_logits = self.word_fc(h_w)
                sentence_word_logits.append(word_logits)
                
                # Teacher Forcing hoặc Greedy
                if target_words is not None:
                    prev_word = target_words[:, i, t]
                else:
                    prev_word = word_logits.argmax(dim=1)
            
            all_word_logits.append(torch.stack(sentence_word_logits, dim=1))

        # Stack kết quả: (B, max_sentences, ...)
        return torch.stack(all_policy_logits, dim=1), \
               torch.stack(all_stop_logits, dim=1), \
               torch.stack(all_word_logits, dim=1)

    def generate(self, image, old_image=None, vocab=None, max_sentences=6, max_words=15, beam_size=3):
        """
        Duyệt qua 2 ảnh để tạo báo cáo so sánh.
        """
        if vocab is None:
            raise ValueError("Vocab must be provided for generation.")

        self.eval()
        device = image.device
        batch_size = image.size(0)
        
        if old_image is None:
            old_image = image

        with torch.no_grad():
            # 1. Feature extraction
            spatial_new = self.image_encoder(image) 
            global_new = self.get_global_features(spatial_new)
            
            spatial_old = self.image_encoder(old_image)
            global_old = self.get_global_features(spatial_old)

            global_combined = torch.cat([global_new, global_old], dim=1)
            global_fused = self.global_fusion(global_combined)

            # Flatten & Permute Spatial
            if spatial_new.dim() == 4:
                # (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
                sn_flat = spatial_new.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, self.encoder_dim)
                so_flat = spatial_old.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, self.encoder_dim)
            else:
                sn_flat = spatial_new.reshape(batch_size, -1, self.encoder_dim)
                so_flat = spatial_old.reshape(batch_size, -1, self.encoder_dim)
            
            spatial_features_flat = torch.cat([sn_flat, so_flat], dim=1)
            
            # 2. Init Topic state
            h_s = self.init_hidden_state(global_combined)
            
            generated_sentences = []
            used_templates = set() # Tránh chọn lại template đã dùng
            
            for i in range(max_sentences):
                h_s = self.sentence_decoder(global_fused, h_s)
                
                # Predict Action & Stop
                policy_logits = self.policy_head(h_s) # (B, 1 + target_templates)
                stop_logits = self.stop_head(h_s)
                
                # --- REPETITION PENALTY: Chặn chọn lại Template đã dùng ---
                for t_idx in used_templates:
                    policy_logits[0, t_idx + 1] -= 100.0 # Index 0 là Generate

                action = policy_logits.argmax(dim=1).item()
                stop = torch.sigmoid(stop_logits).item()
                
                if stop > 0.8: # Ngưỡng dừng linh hoạt hơn
                    break
                
                if action > 0:
                    # CASE: Retrieval (Template)
                    template_text = self.templates[action - 1]
                    generated_sentences.append(template_text)
                    used_templates.add(action - 1)
                else:
                    # CASE: Generation (Word Decoder + Beam Search)
                    h_w = h_s.clone()
                    
                    # Cấu hình Beam Search đơn giản
                    # [score, word_list, hidden_state]
                    beams = [(0.0, [1], h_w)] # 1 là <start>
                    
                    for t in range(max_words):
                        new_beams = []
                        for score, word_list, prev_h in beams:
                            if word_list[-1] == 2: # <end>
                                new_beams.append((score, word_list, prev_h))
                                continue
                                
                            prev_word = torch.tensor([word_list[-1]], device=device)
                            embeddings = self.word_embedding(prev_word)
                            context, _ = self.attention(spatial_features_flat, prev_h)
                            
                            next_h = self.word_decoder(torch.cat([embeddings, context], dim=1), prev_h)
                            logits = self.word_fc(next_h)
                            log_probs = F.log_softmax(logits, dim=1)
                            
                            # Lấy top k từ tiếp theo
                            topk_probs, topk_id = log_probs.topk(beam_size)
                            
                            for k in range(beam_size):
                                next_score = score + topk_probs[0, k].item()
                                next_word_list = word_list + [topk_id[0, k].item()]
                                new_beams.append((next_score, next_word_list, next_h))
                        
                        # Giữ lại top k chùm tốt nhất
                        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
                        
                        # Nếu tất cả các chùm đều kết thúc bằng <end>, dừng sớm
                        if all(b[1][-1] == 2 for b in beams):
                            break
                    
                    # Chọn chùm tốt nhất (bỏ <start> ở đầu)
                    best_words = beams[0][1][1:]
                    # Chuyển ID thành chữ, dừng khi gặp <end>
                    final_words = []
                    for w_idx in best_words:
                        if w_idx == 2: # <end>
                            break
                        final_words.append(vocab.idx2word.get(w_idx, '<unk>'))
                    
                    # Bỏ qua nếu câu rỗng hoặc toàn <unk>
                    if len(final_words) > 0:
                        generated_sentences.append(" ".join(final_words))
            
            return ". ".join(generated_sentences) + "."
