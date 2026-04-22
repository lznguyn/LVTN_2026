import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2), # <--- Thêm Dropout sau GELU
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2)  # <--- Thêm Dropout lớp cuối
        )

    def forward(self, x):
        return self.mlp(x)
