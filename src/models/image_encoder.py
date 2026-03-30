import torch
import torch.nn as nn
import timm

class SwinTransformerV2Encoder(nn.Module):
    def __init__(self, model_name='swinv2_base_window12to16_192to256.22k_ft_in1k', pretrained=True):
        super().__init__()
        # Load pre-trained Swin Transformer V2 (num_classes=0 bỏ qua lớp Classification layer để lấy vector đặc trưng)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.model.num_features

    def forward(self, x):
        # x: (Batch, 3, H, W)
        features = self.model(x)
        return features
