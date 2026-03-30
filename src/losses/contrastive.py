import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringGuidedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds, cluster_ids):
        """
        image_embeds: (Batch, D) - Đặc trưng ảnh đã chuẩn hóa L2
        text_embeds: (Batch, D)  - Đặc trưng text đã chuẩn hóa L2
        cluster_ids: (Batch,)    - Mảng chứa nhãn ID cụm bệnh cho mỗi record
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # Tính toán ma trận Tương đồng (Cosine Similarity Matrix) cho toàn bộ batch
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        # Tạo mask để phát hiện False Negatives: mask[i, j] = 1 nếu cùng nhãn cụm
        cluster_ids = cluster_ids.view(-1, 1)
        cluster_mask = torch.eq(cluster_ids, cluster_ids.t()).float().to(device)
        
        # Positive mask là đường chéo (True Positive pairs)
        positive_mask = torch.eye(batch_size).to(device)
        
        # False Negative mask: Cùng cụm bệnh nhưng là các index khác nhau (Bỏ qua đường chéo)
        false_negative_mask = cluster_mask - positive_mask
        
        # Áp dụng masking: Set logits của false negatives thành 1 số cực nhỏ (-1e9)
        # Các mẫu này sẽ không đóng góp vào gradient đẩy ở mẫu số hàm Softmax (giảm thiểu hình phạt đối với cụm)
        logits_per_image = logits_per_image.masked_fill(false_negative_mask.bool(), -1e9)
        logits_per_text = logits_per_text.masked_fill(false_negative_mask.bool(), -1e9)
        
        # Target cho hàm cross-entropy vẫn là index của đường chéo chính [0, 1, ..., N-1]
        labels = torch.arange(batch_size, dtype=torch.long, device=device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i + loss_t) / 2
        return loss
