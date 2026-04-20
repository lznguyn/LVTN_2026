import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
import argparse
import yaml

# Gắn thư mục gốc vào đường dẫn hệ thống để gọi các modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.data.dataset import MedicalImageTextDataset
from scripts.evaluate import evaluate_retrieval, fix_state_dict

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)

def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def patch_path(p):
    """Sửa lỗi đường dẫn Windows/Linux/Kaggle"""
    if not isinstance(p, str): return p
    p = p.replace('\\', '/')
    # Tìm đoạn 'data/raw/images/' và lấy phần sau nó
    if 'data/raw/images/' in p:
        prefix = 'data/raw/images/'
        return prefix + p.split(prefix)[-1]
    return p

def plot_tsne(img_embeds, txt_embeds, clusters, output_path="results/tsne_clusters.png", lang='vi'):
    """Vẽ biểu đồ t-SNE phân cụm"""
    print("🚀 Đang chạy t-SNE (có thể mất vài phút)...")
    
    # Gộp ảnh và văn bản để xem tương quan (Optional)
    # Ở đây chúng ta chỉ vẽ văn bản trước để thấy cấu trúc bệnh lý
    X = txt_embeds.numpy()
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Lọc bỏ cluster -1 (nhiễu)
    valid_idx = clusters != -1
    X_plot = X_2d[valid_idx]
    c_plot = clusters[valid_idx]
    
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=c_plot, cmap='tab20', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Cluster ID')
    
    title = "Phân cụm không gian đặc trưng (t-SNE)" if lang == 'vi' else "Feature Space Clustering (t-SNE)"
    xlabel = "Chiều t-SNE 1" if lang == 'vi' else "t-SNE Dimension 1"
    ylabel = "Chiều t-SNE 2" if lang == 'vi' else "t-SNE Dimension 2"
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Biểu đồ t-SNE đã được lưu tại: {output_path}")

def plot_retrieval_samples(model, dataloader, device, output_dir="results/retrieval_samples", num_samples=5, lang='vi'):
    """Vẽ demo truy xuất thực tế"""
    print(f"🚀 Đang tạo {num_samples} mẫu truy xuất demo...")
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    all_img_embeds = []
    all_txt_embeds = []
    all_reports = []
    all_images_raw = [] # Keep original images for display
    
    # 1. Trích xuất toàn bộ embeddings
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Samples"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            img_embeds, txt_embeds = model(images, input_ids, attention_mask)
            
            all_img_embeds.append(img_embeds.cpu())
            all_txt_embeds.append(txt_embeds.cpu())
            all_reports.extend(batch['raw_report'])
            
            if len(all_images_raw) < 50: # Chỉ lưu một xíu để chọn mẫu
                # Convert tensor back to image for plotting (un-normalize)
                inv_norm = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                for i in range(images.size(0)):
                   img = inv_norm(images[i].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
                   all_images_raw.append(img)

    img_embeds = torch.cat(all_img_embeds, dim=0)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)
    
    # 2. Chọn ngẫu nhiên mẫu ảnh
    indices = np.random.choice(range(len(all_images_raw)), num_samples, replace=False)
    
    # 3. Tính tương quan
    sims = torch.matmul(img_embeds[indices], txt_embeds.t())
    
    for i, idx in enumerate(indices):
        top_k = torch.topk(sims[i], 3).indices.tolist()
        
        plt.figure(figsize=(15, 8))
        
        # Plot Image
        plt.subplot(1, 2, 1)
        plt.imshow(all_images_raw[idx])
        plt.title("Query X-Ray", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Plot Text Info
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        gt_report = all_reports[idx]
        text_content = f"**[ORIGINAL REPORT]**\n{gt_report[:200]}...\n\n"
        text_content += "**[RETRIEVED REPORTS (TOP 3)]**\n"
        
        for rank, r_idx in enumerate(top_k):
            text_content += f"{rank+1}. {all_reports[r_idx][:150]}...\n\n"
        
        plt.text(0, 1, text_content, transform=plt.gca().transAxes, 
                 verticalalignment='top', fontsize=10, linespacing=1.5, wrap=True)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"✅ Các mẫu truy xuất đã được lưu vào: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Thesis Visualization Tool')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--mode', type=str, default='all', choices=['tsne', 'retrieval', 'all'])
    parser.add_argument('--lang', type=str, default='vi', choices=['vi', 'en'])
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Resources
    print(f"📦 Đang khởi tạo Model từ: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    transform = get_transforms(config['data']['image_size'])
    
    model = MultimodalModel(config['model']['image_encoder'], config['model']['text_encoder']).to(device)
    
    # Load weights (handle Kaggle path and state_dict wrapping)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    state_dict = fix_state_dict(state_dict, model.state_dict().keys())
    model.load_state_dict(state_dict)
    
    # 2. Load Data
    print("📂 Đang nạp dữ liệu Validation...")
    val_df = pd.read_csv(config['data']['val_csv'])
    val_df['image_path'] = val_df['image_path'].apply(patch_path)
    
    # Check if we are on Kaggle and adjust data root if needed
    if os.path.exists("/kaggle/input"):
        # Giả định dữ liệu nằm trong /kaggle/input/dataset-name/...
        # User sẽ cần chỉnh đường dẫn trong config hoặc chúng ta tự quét
        pass

    val_dataset = MedicalImageTextDataset(val_df, transform, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 3. Thực hiện trực quan hóa
    if args.mode in ['tsne', 'all']:
        img_embeds, txt_embeds, clusters = evaluate_retrieval(model, val_loader, device, return_embeds=True)
        plot_tsne(img_embeds, txt_embeds, clusters, lang=args.lang)
        
    if args.mode in ['retrieval', 'all']:
        plot_retrieval_samples(model, val_loader, device, num_samples=5, lang=args.lang)

if __name__ == "__main__":
    main()
