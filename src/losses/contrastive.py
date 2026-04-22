import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringGuidedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, fn_penalty=5.0):
        super().__init__()
        self.temperature = temperature
        self.fn_penalty = fn_penalty # <--- Cho phép cấu hình mức phạt False Negatives

    def forward(self, image_embeds, text_embeds, cluster_ids, soft_labels=None):
        """
        image_embeds: (Batch, D) - Đặc trưng ảnh đã chuẩn hóa L2
        text_embeds: (Batch, D)  - Đặc trưng text đã chuẩn hóa L2
        cluster_ids: (Batch,)    - Mảng chứa nhãn ID cụm bệnh (Hard)
        soft_labels: (Batch, K)  - Ma trận xác suất cụm mềm (GMM)
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 1. Tính toán ma trận Tương đồng Cosine cơ bản
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        # 2. Xử lý Trọng số Mềm (Soft Masking)
        if soft_labels is not None:
            # Tính độ chồng lấn ngữ nghĩa giữa các mẫu trong batch: (B, K) @ (K, B) -> (B, B)
            # w[i, j] càng cao thì i và j càng có chung nhiều nhóm bệnh
            semantic_overlap = torch.matmul(soft_labels, soft_labels.t())
            
            # Chúng ta giữ lại đường chéo (Positive = 1.0) và quan tâm đến các False Negatives
            positive_mask = torch.eye(batch_size).to(device)
            # Trọng số mềm cho các cặp khác nhau: w_ij = overlap nếu i != j, else 0
            soft_fn_mask = semantic_overlap * (1 - positive_mask)
            
            # --- CƠ CHẾ SOFT MASKING (Phiên bản Luận văn - Task 3) ---
            # Giảm penalty factor từ 10.0 xuống 5.0 để tránh làm tác vụ học quá "dễ"
            logits_per_image = logits_per_image - (soft_fn_mask * self.fn_penalty)
            logits_per_text = logits_per_text - (soft_fn_mask * self.fn_penalty)
        else:
            # Fallback về Hard Masking nếu không có nhãn mềm
            cluster_ids = cluster_ids.view(-1, 1)
            cluster_mask = torch.eq(cluster_ids, cluster_ids.t()).float().to(device)
            positive_mask = torch.eye(batch_size).to(device)
            false_negative_mask = cluster_mask - positive_mask
            # Sử dụng fn_penalty thay vì fix cứng -10000 (nếu muốn) 
            # Nhưng ở chế độ Hard, ta vẫn nên giữ mask mạnh
            logits_per_image = logits_per_image.masked_fill(false_negative_mask.bool(), -10000.0)
            logits_per_text = logits_per_text.masked_fill(false_negative_mask.bool(), -10000.0)
        
        # Target cho hàm cross-entropy vẫn là index của đường chéo chính [0, 1, ..., N-1]
        labels = torch.arange(batch_size, dtype=torch.long, device=device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i + loss_t) / 2
        return loss
