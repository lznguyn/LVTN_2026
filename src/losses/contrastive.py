import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringGuidedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, fn_penalty=3.0, feature_dim=768, queue_size=1024):
        super().__init__()
        self.temperature = temperature
        self.fn_penalty = fn_penalty # <--- Giảm xuống 3.0 để Model phải phân biệt gắt hơn giữa các mẫu tương đồng
        self.queue_size = queue_size
        self.feature_dim = feature_dim

        # Khởi tạo Memory Bank (không tính gradient, đăng ký làm buffer để tự lưu vào state_dict)
        self.register_buffer("image_queue", torch.randn(feature_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(feature_dim, queue_size))
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        
        # Hàng đợi nhãn cụm Hard (Mặc định -1)
        self.register_buffer("cluster_queue", -1 * torch.ones(queue_size, dtype=torch.long))
        
        # Con trỏ chỉ vị trí của queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Hàng đợi nhãn mềm (Sẽ được cấp phát lười - lazy init khi chạy để tự căn num_clusters)
        self.soft_label_queue = None

    @torch.no_grad()
    def _dequeue_and_enqueue(self, images, texts, cluster_ids, soft_labels=None):
        """Cập nhật dữ liệu vào hàng đợi (Memory Bank)"""
        batch_size = images.shape[0]
        ptr = int(self.queue_ptr)
        
        # Xử lý trường hợp nhét vào queue bị dư
        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr
            images = images[:batch_size]
            texts = texts[:batch_size]
            cluster_ids = cluster_ids[:batch_size]
            if soft_labels is not None:
                soft_labels = soft_labels[:batch_size]

        # Đẩy vào queue (Transpose để về dạng Feature x Size)
        self.image_queue[:, ptr:ptr + batch_size] = images.T
        self.text_queue[:, ptr:ptr + batch_size] = texts.T
        self.cluster_queue[ptr:ptr + batch_size] = cluster_ids
        
        if soft_labels is not None and hasattr(self, 'soft_label_queue') and self.soft_label_queue is not None:
            self.soft_label_queue[:, ptr:ptr + batch_size] = soft_labels.T
            
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, image_embeds, text_embeds, cluster_ids, soft_labels=None):
        """
        image_embeds: (Batch, D) - Đặc trưng ảnh đã chuẩn hóa L2
        text_embeds: (Batch, D)  - Đặc trưng text đã chuẩn hóa L2
        cluster_ids: (Batch,)    - Mảng chứa nhãn ID cụm bệnh (Hard)
        soft_labels: (Batch, K)  - Ma trận xác suất cụm mềm (GMM)
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # [LAZY INIT] Khởi tạo hàng đợi soft labels nếu chưa có
        if soft_labels is not None and (not hasattr(self, 'soft_label_queue') or self.soft_label_queue is None):
            num_clusters = soft_labels.size(1)
            self.soft_label_queue = torch.zeros(num_clusters, self.queue_size, device=device)
        
        # Đảm bảo hàng đợi soft_label (nếu có) luôn ở đúng device
        if hasattr(self, 'soft_label_queue') and self.soft_label_queue is not None and self.soft_label_queue.device != device:
            self.soft_label_queue = self.soft_label_queue.to(device)
        
        # 1. Tương đồng với mẫu trong cùng Batch (In-batch Positives & Negatives)
        l_pos_neg_i = torch.matmul(image_embeds, text_embeds.t())
        l_pos_neg_t = l_pos_neg_i.t()
        
        # 2. Tương đồng với hàng ngàn mẫu trong Queue (Khắc phục lỗi batch_size=8)
        # clone().detach() để chắc chắn gradient không truyền qua hàng đợi
        l_queue_i = torch.matmul(image_embeds, self.text_queue.clone().detach())
        l_queue_t = torch.matmul(text_embeds, self.image_queue.clone().detach())
        
        # Gộp In-batch và Queue => Ma trận kích thước (Batch, Batch + Queue)
        logits_per_image = torch.cat([l_pos_neg_i, l_queue_i], dim=1) / self.temperature
        logits_per_text = torch.cat([l_pos_neg_t, l_queue_t], dim=1) / self.temperature
        
        # 3. Xử lý Trọng số Mềm (Soft Masking)
        if soft_labels is not None:
            # Mask cho các mẫu In-batch
            semantic_overlap_batch = torch.matmul(soft_labels, soft_labels.t())
            positive_mask = torch.eye(batch_size).to(device)
            soft_fn_mask_batch = semantic_overlap_batch * (1 - positive_mask)
            
            # Mask cho các mẫu trong Queue
            semantic_overlap_queue = torch.matmul(soft_labels, self.soft_label_queue.clone().detach())
            
            # Gộp lại
            soft_fn_mask = torch.cat([soft_fn_mask_batch, semantic_overlap_queue], dim=1)
            
            # Trừ penalty: Càng giống nhau (soft label overlap cao) thì bị đẩy xa ít đi
            logits_per_image = logits_per_image - (soft_fn_mask * self.fn_penalty)
            logits_per_text = logits_per_text - (soft_fn_mask * self.fn_penalty)
        else:
            # Fallback về Hard Masking nếu không có nhãn mềm
            cluster_ids = cluster_ids.view(-1, 1)
            
            # Xử lý In-batch
            cluster_mask_batch = torch.eq(cluster_ids, cluster_ids.t()).float().to(device)
            positive_mask = torch.eye(batch_size).to(device)
            fn_mask_batch = cluster_mask_batch - positive_mask
            
            # Xử lý Queue
            queue_ids = self.cluster_queue.clone().detach().view(1, -1)
            fn_mask_queue = torch.eq(cluster_ids, queue_ids).float().to(device)
            
            # Gộp lại
            fn_mask = torch.cat([fn_mask_batch, fn_mask_queue], dim=1)
            
            # Đẩy xa vô cực các True Negatives (trùng cụm)
            logits_per_image = logits_per_image.masked_fill(fn_mask.bool(), -10000.0)
            logits_per_text = logits_per_text.masked_fill(fn_mask.bool(), -10000.0)
        
        # Target cho hàm cross-entropy vẫn là index của đường chéo chính [0, 1, ..., N-1]
        # (Vì Positives chỉ nằm ở In-batch chéo)
        labels = torch.arange(batch_size, dtype=torch.long, device=device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i + loss_t) / 2
        
        # 4. Cập nhật Queue cho lần lặp tiếp theo
        # Truyền vào detach() để ngắt hoàn toàn đạo hàm
        self._dequeue_and_enqueue(
            image_embeds.detach(), 
            text_embeds.detach(), 
            cluster_ids.detach(), 
            soft_labels.detach() if soft_labels is not None else None
        )
        
        return loss
