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

    # Chuyển thành Tensor khổng lồ
    img_embeds = torch.cat(all_img_embeds, dim=0)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)
    
    num_samples = img_embeds.size(0)
    print(f"✅ Đã trích xuất xong {num_samples} mẫu.")

    print("\n--- [2] TÍNH TOÁN MA TRẬN TƯƠNG ĐỒNG (SIMILARITY MATRIX) ---")
    # Di chuyển lên GPU để tính toán cho nhanh (Nếu dữ liệu quá lớn > 50k mẫu thì dùng CPU hoặc Chunking)
    img_embeds = img_embeds.to(device)
    txt_embeds = txt_embeds.to(device)
    
    # Tính tích vô hướng (Dot product) vì vector đã được chuẩn hóa L2 -> kết quả là Cosine Similarity
    sim_matrix = torch.matmul(img_embeds, txt_embeds.t())
    
    # --- IMAGE-TO-TEXT (I2T) RETRIEVAL ---
    print("\n--- [3] KẾT QUẢ IMAGE-TO-TEXT (ẢNH TÌM VĂN BẢN) ---")
    i2t_r1, i2t_r5, i2t_r10 = calculate_recall(sim_matrix, axis=1)
    
    # --- TEXT-TO-IMAGE (T2I) RETRIEVAL ---
    print("\n--- [4] KẾT QUẢ TEXT-TO-IMAGE (VĂN BẢN TÌM ẢNH) ---")
    t2i_r1, t2i_r5, t2i_r10 = calculate_recall(sim_matrix, axis=0)

    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ CUỐI CÙNG (RECALL @ K)")
    print("="*50)
    print(f"🔹 Image-to-Text: R@1: {i2t_r1:.2f}% | R@5: {i2t_r5:.2f}% | R@10: {i2t_r10:.2f}%")
    print(f"🔹 Text-to-Image: R@1: {t2i_r1:.2f}% | R@5: {t2i_r5:.2f}% | R@10: {t2i_r10:.2f}%")
    print("="*50)

    # In ra 3 ví dụ demo
    print("\n👁️ VÍ DỤ TRUY VẤN DEMO (I2T):")
    for i in range(3):
        idx = np.random.randint(0, num_samples)
        top5_indices = torch.topk(sim_matrix[idx], 5).indices.cpu().numpy()
        print(f"\n[Mẫu {idx}] Văn bản gốc: {all_reports[idx][:100]}...")
        print(f"Top 1 Dự đoán: {all_reports[top5_indices[0]][:100]}...")
        if top5_indices[0] == idx:
            print("✅ KẾT QUẢ: TÌM THẤY CHÍNH XÁC!")
        else:
            print("❌ KẾT QUẢ: KHÔNG KHỚP TOP 1.")

def calculate_recall(sim_matrix, axis=1):
    """
    sim_matrix: (N, N)
    axis=1: Image looking for text
    axis=0: Text looking for image
    """
    if axis == 0:
        sim_matrix = sim_matrix.t()
    
    batch_size = sim_matrix.size(0)
    # Tìm index của top K giá trị lớn nhất trong mỗi hàng
    top10_indices = torch.topk(sim_matrix, 10, dim=1).indices
    
    # Index thật (Ground Truth) chính là [0, 1, 2, ..., N-1] nằm trên đường chéo
    targets = torch.arange(batch_size, device=sim_matrix.device).view(-1, 1)
    
    # Kiểm tra xem target có nằm trong top K không
    r1 = (top10_indices[:, :1] == targets).any(dim=1).float().mean().item() * 100
    r5 = (top10_indices[:, :5] == targets).any(dim=1).float().mean().item() * 100
    r10 = (top10_indices[:, :10] == targets).any(dim=1).float().mean().item() * 100
    
    return r1, r5, r10

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
