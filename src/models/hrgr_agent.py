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
        
        # Khởi tạo hidden state từ Image Global Features
        self.init_h = nn.Linear(self.encoder_dim, decoder_dim)

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

    def forward(self, images, target_actions=None, target_words=None, max_sentences=8, max_words=20):
        """
        Forward pass hỗ trợ cả Training (MLE) và Inference.
        """
        batch_size = images.size(0)
        device = images.device

        # 1. Image features
        spatial_features = self.image_encoder(images) # (B, H, W, C)
        global_features = self.get_global_features(spatial_features)
        
        # Flatten spatial: (B, NumPixels, C)
        # Nếu đã là (B, H, W, C), chỉ cần reshape thành (B, H*W, C)
        if spatial_features.dim() == 4:
             spatial_features = spatial_features.reshape(batch_size, -1, self.encoder_dim)
        else:
             spatial_features = spatial_features.permute(0, 2, 3, 1).reshape(batch_size, -1, self.encoder_dim)
        
        # 2. Init Sentence Decoder
        h_s = self.init_hidden_state(global_features)
        
        all_policy_logits = []
        all_stop_logits = []
        all_word_logits = [] # (B, max_sentences, max_words, vocab_size)

        for i in range(max_sentences):
            # Cập nhật topic vector q_i (h_s)
            h_s = self.sentence_decoder(global_features, h_s)
            
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

    def generate(self, image, vocab, max_sentences=6, max_words=15):
        """
        Dùng cho Inference: Nhận 1 ảnh, sinh ra báo cáo hoàn chỉnh.
        """
        self.eval()
        device = image.device
        batch_size = image.size(0)

        with torch.no_grad():
            # 1. Feature extraction
            spatial_features = self.image_encoder(image)
            global_features = self.get_global_features(spatial_features)
            
            # Flatten spatial: (B, NumPixels, C)
            if spatial_features.dim() == 4:
                spatial_features = spatial_features.reshape(batch_size, -1, self.encoder_dim)
            else:
                spatial_features = spatial_features.permute(0, 2, 3, 1).reshape(batch_size, -1, self.encoder_dim)
            
            # 2. Init Topic state
            h_s = self.init_hidden_state(global_features)
            
            generated_sentences = []
            
            for i in range(max_sentences):
                h_s = self.sentence_decoder(global_features, h_s)
                
                # Predict Action & Stop
                policy_logits = self.policy_head(h_s)
                stop_logits = self.stop_head(h_s)
                
                action = policy_logits.argmax(dim=1).item()
                stop = torch.sigmoid(stop_logits).item()
                
                if stop > 0.5:
                    break
                
                if action > 0:
                    # Lấy từ Template Database
                    template_text = self.templates[action - 1]
                    generated_sentences.append(template_text)
                else:
                    # Tự viết câu mới (Word Decoder)
                    h_w = h_s
                    prev_word = torch.tensor([1], device=device) # <start>
                    sentence_words = []
                    
                    for t in range(max_words):
                        embeddings = self.word_embedding(prev_word)
                        context, _ = self.attention(spatial_features, h_w)
                        h_w = self.word_decoder(torch.cat([embeddings, context], dim=1), h_w)
                        
                        word_logits = self.word_fc(h_w)
                        word_idx = word_logits.argmax(dim=1).item()
                        
                        if word_idx == 2: # <end>
                            break
                        
                        word = vocab.idx2word.get(word_idx, '<unk>')
                        sentence_words.append(word)
                        prev_word = torch.tensor([word_idx], device=device)
                        
                    generated_sentences.append(" ".join(sentence_words))
            
            return ". ".join(generated_sentences) + "."
