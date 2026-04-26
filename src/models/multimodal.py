import torch
import torch.nn as nn
import numpy as np
from .image_encoder import SwinTransformerV2Encoder
from .text_encoder import MedicalTextEncoder
from .projection import ProjectionHead

class MultimodalModel(nn.Module):
    def __init__(self, image_encoder_name, text_model_name, embed_dim=512, image_size=None):
        super().__init__()
        self.image_encoder = SwinTransformerV2Encoder(model_name=image_encoder_name, img_size=image_size)
        self.text_model_name = text_model_name # Lưu lại để dùng nếu cần
        self.text_encoder = MedicalTextEncoder(model_name=text_model_name)
        
        # Ánh xạ cả 2 không gian đặc trưng về cùng 1 số chiều (embed_dim = d)
        self.image_proj = ProjectionHead(self.image_encoder.feature_dim, embed_dim)
        self.text_proj = ProjectionHead(self.text_encoder.feature_dim, embed_dim)
        
        # Learnable Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.embed_dim = embed_dim

    def forward(self, images, text_input_ids, text_attention_mask):
        img_features = self.image_encoder(images)
        txt_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        img_embeds = self.image_proj(img_features)
        txt_embeds = self.text_proj(txt_features)
        
        # Chuẩn hóa L2 norm trước khi tính Cosine Similarity ở loss function
        img_embeds = nn.functional.normalize(img_embeds, dim=-1)
        txt_embeds = nn.functional.normalize(txt_embeds, dim=-1)
        
        return img_embeds, txt_embeds, self.logit_scale
