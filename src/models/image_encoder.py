import torch
import torch.nn as nn
import timm

class SwinTransformerV2Encoder(nn.Module):
    def __init__(
        self,
        model_name='swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
        pretrained=True
    ):
        super().__init__()

        # num_classes=0 => bỏ head classification, lấy feature embedding
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        self.feature_dim = self.model.num_features

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.model(x)
        return features