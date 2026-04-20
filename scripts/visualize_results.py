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

def get_cluster_keywords(df, cluster_id, top_n=3):
    """Trích xuất từ khóa bệnh lý tiêu biểu cho mỗi cụm"""
    from collections import Counter
    import re
    
    # Danh sách stop-words y tế và tiếng Anh cơ bản
    stops = {
        'the', 'and', 'with', 'is', 'no', 'of', 'for', 'was', 'were', 'within', 'are', 'identified', 
        'seen', 'shown', 'appear', 'there', 'from', 'this', 'that', 'these', 'those', 'normal',
        'acute', 'finding', 'findings', 'patient', 'chest', 'lungs', 'pleural', 'heart', 'clear'
    }
    
    text = " ".join(df[df['cluster_id'] == cluster_id]['report'].astype(str).tolist()).lower()
    words = re.findall(r'\b[a-z]{4,}\b', text) # Chỉ lấy từ 4 chữ cái trở lên
    words = [w for w in words if w not in stops]
    
    most_common = [w[0].capitalize() for w in Counter(words).most_common(top_n)]
    return " & ".join(most_common) if most_common else f"Cluster {cluster_id}"

def plot_tsne(img_embeds, txt_embeds, clusters, df=None, output_path="results/tsne_labeled.png", lang='vi'):
    """Vẽ biểu đồ t-SNE phân cụm với nhãn bệnh lý tự động"""
    print("🚀 Đang chạy t-SNE và trích xuất nhãn bệnh lý...")
    
    X = txt_embeds.numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(14, 11))
    sns.set_style("whitegrid")
    
    valid_idx = clusters != -1
    X_plot = X_2d[valid_idx]
    c_plot = clusters[valid_idx]
    
    # Lấy danh sách các cụm duy nhất (loại bỏ -1)
    unique_clusters = sorted(np.unique(c_plot))
    
    # Tạo bảng màu
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = c_plot == cluster_id
        cluster_name = get_cluster_keywords(df, cluster_id) if df is not None else f"Cluster {cluster_id}"
        
        plt.scatter(X_plot[mask, 0], X_plot[mask, 1], label=cluster_name, alpha=0.6, s=25)
        
        # Thêm Annotation tại vị trí trung tâm cụm (Centroid)
        centroid = X_plot[mask].mean(axis=0)
        plt.annotate(cluster_name, centroid, fontsize=9, fontweight='bold', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    title = "Phân biệt không gian đặc trưng theo Bệnh lý (t-SNE)" if lang == 'vi' else "Pathology-aware Feature Space (t-SNE)"
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.legend(title="Nhóm bệnh lý (Auto-labeled)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Biểu đồ t-SNE gán nhãn đã được lưu tại: {output_path}")

def plot_joint_tsne(img_embeds, txt_embeds, clusters, output_path="results/joint_space.png", lang='vi'):
    """Vẽ biểu đồ Joint Space (Ảnh + Văn bản) để xem độ Alignment"""
    print("🚀 Đang chạy Joint t-SNE (Ảnh + Văn bản)...")
    
    # Gộp cả image và text embeddings
    # Dùng 1000 mẫu đầu tiên để biểu đồ không quá bị nhiễu
    N = min(500, len(img_embeds))
    X = torch.cat([img_embeds[:N], txt_embeds[:N]], dim=0).numpy()
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    
    X_img = X_2d[:N]
    X_txt = X_2d[N:]
    c_plot = clusters[:N]
    
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Vẽ các đường nối giữa Ảnh và Văn bản của cùng một bệnh nhân (Alignment)
    for i in range(N):
        plt.plot([X_img[i, 0], X_txt[i, 0]], [X_img[i, 1], X_txt[i, 1]], 'gray', alpha=0.1, linewidth=0.5)
    
    plt.scatter(X_img[:, 0], X_img[:, 1], c=c_plot, marker='o', s=30, alpha=0.7, label='Image Embeds', cmap='tab20')
    plt.scatter(X_txt[:, 0], X_txt[:, 1], c=c_plot, marker='^', s=40, alpha=0.8, label='Text Embeds', cmap='tab20')
    
    plt.title("Sự căn chỉnh không gian đa phương thức (Joint Alignment)", fontsize=16, fontweight='bold')
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Biểu đồ Joint Space đã được lưu tại: {output_path}")

def plot_comparative_t2i(config, ckpt_a, ckpt_b, query_text, val_loader, device, output_path="results/comparative_t2i.png"):
    """Vẽ biểu đồ so sánh Top-3 ảnh giữa 2 mô hình cho cùng 1 câu query"""
    print(f"🚀 Đang so sánh truy xuất cho query: '{query_text}'")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    def get_top_k_for_model(ckpt_path, query_str, loader, k=3):
        # Khởi tạo model
        model = MultimodalModel(config['model']['image_encoder'], config['model']['text_encoder']).to(device)
        
        # Nếu có ckpt_path thì mới nạp weights, nếu không sẽ là Untrained Baseline
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = fix_state_dict(sd, model.state_dict().keys())
            model.load_state_dict(sd)
            print(f"📦 Đã nạp weights từ: {os.path.basename(ckpt_path)}")
        else:
            print("📦 Sử dụng mô hình chưa huấn luyện (Untrained Baseline)")
            
        model.eval()
        
        # 1. Encode Query Text
        inputs = tokenizer(query_str, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            _, query_embed = model(None, inputs['input_ids'], inputs['attention_mask'], text_only=True)
            
        # 2. Encode Images
        all_img_embeds = []
        all_images_raw = []
        desc = f"Encoding images ({os.path.basename(ckpt_path) if ckpt_path else 'Untrained'})"
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                imgs = batch['image'].to(device)
                img_embeds, _ = model(imgs, None, None, image_only=True)
                all_img_embeds.append(img_embeds.cpu())
                
                if len(all_images_raw) < 100: 
                    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                    for j in range(imgs.size(0)):
                        all_images_raw.append(inv_norm(imgs[j].cpu()).clamp(0, 1).permute(1, 2, 0).numpy())
                        
        img_embeds = torch.cat(all_img_embeds, dim=0)
        sims = torch.matmul(query_embed.cpu(), img_embeds.t()) 
        top_vals, top_idx = torch.topk(sims[0], k)
        
        return top_idx.tolist(), top_vals.tolist(), all_images_raw

    # Lấy kết quả cho 2 model
    idx_a, vals_a, imgs_raw = get_top_k_for_model(ckpt_a, query_text, val_loader)
    idx_b, vals_b, _        = get_top_k_for_model(ckpt_b, query_text, val_loader)
    
    # 3. Vẽ biểu đồ so sánh
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ckpt_a_name = os.path.basename(ckpt_a) if ckpt_a else "Untrained Baseline"
    ckpt_b_name = os.path.basename(ckpt_b) if ckpt_b else "SOTA"
    
    plt.suptitle(f"Query: \"{query_text}\"\n(Comparative Inference: {ckpt_a_name} vs {ckpt_b_name})", 
                 fontsize=16, fontweight='bold')
    
    labels = [ckpt_a_name, ckpt_b_name]
    all_indices = [idx_a, idx_b]
    all_values = [vals_a, vals_b]
    
    for row in range(2):
        for col in range(3):
            idx = all_indices[row][col]
            sim = all_values[row][col]
            
            axes[row, col].imshow(imgs_raw[idx])
            axes[row, col].set_title(f"{labels[row]} - Rank {col+1}\nSim: {sim:.3f}", fontsize=12)
            axes[row, col].axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"✅ Ảnh so sánh đã được lưu tại: {output_path}")

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
    parser.add_argument('--checkpoint', type=str, help='Path to main .pth model')
    parser.add_argument('--ckpt_a', type=str, help='Path to model A (Baseline)')
    parser.add_argument('--ckpt_b', type=str, help='Path to model B (SOTA)')
    parser.add_argument('--query', type=str, default="Enlarged cardiac silhouette consistent with cardiomegaly.", help='Text query for T2I')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--mode', type=str, default='all', choices=['tsne', 'retrieval', 'compare', 'all'])
    parser.add_argument('--lang', type=str, default='vi', choices=['vi', 'en'])
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data (Dùng chung cho tất cả các mode)
    print("📂 Đang nạp dữ liệu Validation...")
    val_df = pd.read_csv(config['data']['val_csv'])
    val_df['image_path'] = val_df['image_path'].apply(patch_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    transform = get_transforms(config['data']['image_size'])
    val_dataset = MedicalImageTextDataset(val_df, transform, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 2. Xử lý theo từng Mode
    if args.mode == 'compare':
        if not args.ckpt_b:
            print("❌ Lỗi: Chế độ 'compare' yêu cầu ít nhất --ckpt_b (mô hình SOTA)")
            return
        # Nếu không có ckpt_a, script sẽ tự động dùng Untrained Baseline
        plot_comparative_t2i(config, args.ckpt_a, args.ckpt_b, args.query, val_loader, device)
        return

    # Các mode khác yêu cầu nạp 1 model chính
    if not args.checkpoint:
        print("❌ Lỗi: Cần cung cấp --checkpoint cho mode này.")
        return

    print(f"📦 Đang khởi tạo Model từ: {args.checkpoint}")
    model = MultimodalModel(config['model']['image_encoder'], config['model']['text_encoder']).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    state_dict = fix_state_dict(state_dict, model.state_dict().keys())
    model.load_state_dict(state_dict)
    
    if args.mode in ['tsne', 'all']:
        img_embeds, txt_embeds, clusters = evaluate_retrieval(model, val_loader, device, return_embeds=True)
        plot_tsne(img_embeds, txt_embeds, clusters, df=val_df, lang=args.lang)
        plot_joint_tsne(img_embeds, txt_embeds, clusters, lang=args.lang)
        
    if args.mode in ['retrieval', 'all']:
        plot_retrieval_samples(model, val_loader, device, num_samples=5, lang=args.lang)

if __name__ == "__main__":
    main()
