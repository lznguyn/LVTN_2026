import torch
import torch.nn as nn
import timm

class SwinTransformerV2Encoder(nn.Module):
    def __init__(
        self,
        model_name='swinv2_base_window12to24_192to384',
        pretrained=True,
        features_only=False,
        img_size=None
    ):
        super().__init__()

        self.features_only = features_only
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0 if not features_only else None,
            features_only=features_only,
            img_size=img_size
        )
        
        # Bật Gradient Checkpointing để tiết kiệm VRAM, hỗ trợ Batch Size khổng lồ
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(True)
        
        if features_only:
            # Lấy thông tin chiều sâu của Feature Map cuối cùng
            self.feature_dim = self.model.feature_info[-1]['num_chs']
        else:
            self.feature_dim = self.model.num_features

    def forward(self, x):
        # x: (B, 3, H, W)
        if self.features_only:
            # Trả về list các feature maps từ các stage, chọn stage cuối
            feature_maps = self.model(x)
            return feature_maps[-1] # Shape: (B, C, H_f, W_f)
        else:
            features = self.model(x)
            return features