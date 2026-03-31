import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import os
import sys

# Thiết lập đường dẫn hệ thống để gọi 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.data.dataset import MedicalImageTextDataset

def load_config(config_path="configs/default.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)

def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device):
    model.eval()
    all_img_embeds = []
    all_txt_embeds = []
    all_reports = []

    print("--- [1] TRÍCH XUẤT ĐẶC TRƯNG TẬP VALIDATION ---")
    for batch in tqdm(dataloader, desc="Extracting Embeddings"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Chỉ lấy đặc trưng (embedding)
        img_embeds, txt_embeds = model(images, input_ids, attention_mask)
        
        all_img_embeds.append(img_embeds.cpu())
        all_txt_embeds.append(txt_embeds.cpu())
        all_reports.extend(batch['raw_report'])

    img_embeds = torch.cat(all_img_embeds, dim=0)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)
    num_samples = img_embeds.size(0)
    print(f"✅ Đã trích xuất xong {num_samples} mẫu.")

    # Giải phóng bộ nhớ GPU trước khi tính Similarity
    torch.cuda.empty_cache()

    print("\n--- [2] TÍNH TOÁN RECALL@K THEO PHƯƠNG PHÁP CHUNKING (TIẾT KIỆM RAM) ---")
    
    # Image-to-Text (Truy vấn ảnh, lấy văn bản)
    i2t_r1, i2t_r5, i2t_r10 = calculate_recall_chunked(img_embeds, txt_embeds, device, chunk_size=1000)
    
    # Text-to-Image (Truy vấn văn bản, lấy ảnh)
    t2i_r1, t2i_r5, t2i_r10 = calculate_recall_chunked(txt_embeds, img_embeds, device, chunk_size=1000)

    print("\n" + "="*60)
    print(f"📊 KẾT QUẢ CUỐI CÙNG (RECALL @ K)")
    print("="*60)
    print(f"🔹 Image-to-Text (I2T): R@1: {i2t_r1:.2f}% | R@5: {i2t_r5:.2f}% | R@10: {i2t_r10:.2f}%")
    print(f"🔹 Text-to-Image (T2I): R@1: {t2i_r1:.2f}% | R@5: {t2i_r5:.2f}% | R@10: {t2i_r10:.2f}%")
    print("="*60)

    # In 3 mẫu demo (Tận dụng CPU embeds để demo)
    print("\n👁️ VÍ DỤ TRUY VẤN DEMO (I2T):")
    # Chỉ tính sim cho 10 mẫu đầu tiên để trình diễn
    sample_sim = torch.matmul(img_embeds[:10].to(device), txt_embeds.to(device).t())
    for i in range(3):
        top5_indices = torch.topk(sample_sim[i], 5).indices.cpu().numpy()
        print(f"\n[Mẫu {i}] Văn bản gốc: {all_reports[i][:100]}...")
        print(f"Top 1 Dự đoán: {all_reports[top5_indices[0]][:100]}...")
        print("✅ KẾT QUẢ: " + ("CHÍNH XÁC!" if top5_indices[0] == i else "KHÔNG KHỚP TOP 1."))

def calculate_recall_chunked(query_embeds, gallery_embeds, device, chunk_size=1000):
    """
    Tính Recall@K mà không tạo toàn bộ ma trận similarity trên GPU cùng lúc.
    query_embeds: (N, D) - CPU tensor
    gallery_embeds: (M, D) - CPU tensor
    """
    num_queries = query_embeds.size(0)
    num_gallery = gallery_embeds.size(0)
    
    hits_r1, hits_r5, hits_r10 = 0, 0, 0
    
    # Đưa gallery lên GPU một lần (Nếu gallery quá lớn, có thể chunk nốt cả gallery)
    gallery_embeds_gpu = gallery_embeds.to(device).t() # (D, M)

    for i in tqdm(range(0, num_queries, chunk_size), desc="Calculating Recall"):
        start = i
        end = min(i + chunk_size, num_queries)
        
        # Lấy một khối query đưa lên GPU
        query_chunk = query_embeds[start:end].to(device)
        
        # Tính tương đồng cho khối hiện tại: (chunk_size, M)
        sim_chunk = torch.matmul(query_chunk, gallery_embeds_gpu)
        
        # Tìm top 10 cho mỗi query trong gallery
        top10_indices = torch.topk(sim_chunk, 10, dim=1).indices
        
        # Target index chính là index thực của query trong toàn bộ gallery (Giả định 1-1 matching)
        targets = torch.arange(start, end, device=device).view(-1, 1)
        
        hits_r1 += (top10_indices[:, :1] == targets).any(dim=1).sum().item()
        hits_r5 += (top10_indices[:, :5] == targets).any(dim=1).sum().item()
        hits_r10 += (top10_indices[:, :10] == targets).any(dim=1).sum().item()
        
        # Dọn dẹp biến tạm để tiết kiệm RAM GPU
        del sim_chunk
        del top10_indices
        # Optional: torch.cuda.empty_cache() nếu cần thiết hơn nữa

    return (hits_r1/num_queries)*100, (hits_r5/num_queries)*100, (hits_r10/num_queries)*100


def main():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Nạp mô hình
    print(f"--- ĐANG NẠP MÔ HÌNH TỪ CHECKPOINT: {config['training']['checkpoint_dir']} ---")
    model = MultimodalModel(
        image_encoder_name=config['model']['image_encoder'],
        text_model_name=config['model']['text_encoder'],
        embed_dim=config['model']['embed_dim']
    ).to(device)
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ Lỗi: Không tìm thấy file {ckpt_path}. Bạn đã chạy Train chưa?")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # 2. Chuẩn bị dữ liệu
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    image_transform = get_transforms(config['data']['image_size'])
    
    val_df = pd.read_csv(config['data']['val_csv'])
    # Chế độ evaluate: Không shuffle, batch size nên lớn hơn để nhanh
    val_dataset = MedicalImageTextDataset(val_df, image_transform, tokenizer, config['data']['max_text_length'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'] * 2, shuffle=False)
    
    evaluate_retrieval(model, val_loader, device)

if __name__ == "__main__":
    main()
