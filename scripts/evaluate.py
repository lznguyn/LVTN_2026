import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import os
import sys
import argparse
import glob
import json
import re

# Thiết lập đường dẫn hệ thống để gọi 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.models.hrgr_agent import HRGRAgent
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

def fix_state_dict(state_dict, model_keys):
    has_layers_dot = any("layers." in k for k in model_keys)
    has_layers_underscore = any("layers_" in k for k in model_keys)
    is_new_model = any(".mlp.5." in k for k in model_keys)
    is_old_checkpoint = is_new_model and not any(".mlp.5." in k for k in state_dict.keys())
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if has_layers_dot and "layers_" in k:
            new_k = new_k.replace("layers_", "layers.")
        elif has_layers_underscore and "layers." in k:
            new_k = new_k.replace("layers.", "layers_")
        if is_old_checkpoint:
            if ".mlp.4." in new_k:
                new_k = new_k.replace(".mlp.4.", ".mlp.5.")
            elif ".mlp.3." in new_k:
                new_k = new_k.replace(".mlp.3.", ".mlp.4.")
        new_state_dict[new_k] = v
    return new_state_dict

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, return_embeds=False):
    model.eval()
    all_img_embeds, all_txt_embeds, all_clusters = [], [], []

    for batch in tqdm(dataloader, desc="Retrieval Embeddings", leave=False):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        img_embeds, txt_embeds = model(images, input_ids, attention_mask)
        
        all_img_embeds.append(img_embeds.cpu())
        all_txt_embeds.append(txt_embeds.cpu())
        if 'cluster_id' in batch:
            all_clusters.append(batch['cluster_id'].cpu())

    img_embeds = torch.cat(all_img_embeds, dim=0)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)
    clusters = torch.cat(all_clusters, dim=0) if all_clusters and all_clusters[0].min() != -1 else None
    
    if return_embeds:
        return img_embeds, txt_embeds, clusters

    r_strict = calculate_recall_chunked(img_embeds, txt_embeds, device, clusters=None)
    r_cluster = calculate_recall_chunked(img_embeds, txt_embeds, device, clusters=clusters) if clusters is not None else r_strict
    return r_strict, r_cluster

def calculate_recall_chunked(query_embeds, gallery_embeds, device, clusters=None, chunk_size=1000):
    num_queries = query_embeds.size(0)
    hits_r1, hits_r5, hits_r10 = 0, 0, 0
    gallery_embeds_gpu = gallery_embeds.to(device).t()
    if clusters is not None:
        clusters = clusters.to(device)

    for i in range(0, num_queries, chunk_size):
        start = i
        end = min(i + chunk_size, num_queries)
        query_chunk = query_embeds[start:end].to(device)
        sim_chunk = torch.matmul(query_chunk, gallery_embeds_gpu)
        top10_indices = torch.topk(sim_chunk, min(10, sim_chunk.size(1)), dim=1).indices
        targets = torch.arange(start, end, device=device).view(-1, 1)

        if clusters is not None:
            query_clusters = clusters[start:end].unsqueeze(1)
            retrieved_clusters = clusters[top10_indices]
            matches = (top10_indices == targets) | (retrieved_clusters == query_clusters)
        else:
            matches = (top10_indices == targets)

        hits_r1  += matches[:, :1].any(dim=1).sum().item()
        hits_r5  += matches[:, :5].any(dim=1).sum().item()
        hits_r10 += matches[:, :10].any(dim=1).sum().item()

    return (hits_r1/num_queries)*100, (hits_r5/num_queries)*100, (hits_r10/num_queries)*100

def main():
    parser = argparse.ArgumentParser(description='Smart Evaluate Retrieval & Agent')
    parser.add_argument('--checkpoint', type=str, help='Path to .pth file or directory')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--csv', type=str, help='Override dataset CSV path')
    parser.add_argument('--img_dir', type=str, help='Root directory for images')
    parser.add_argument('--output', type=str, default='data/evaluation_summary.csv')
    parser.add_argument('--image_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    img_size = args.image_size if args.image_size else config['data']['image_size']
    transform = get_transforms(img_size)

    # Nạp dữ liệu linh hoạt
    csv_path = args.csv if args.csv else config['data']['val_csv']
    print(f"📖 Đang nạp dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if args.img_dir:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(args.img_dir, x) if not os.path.isabs(x) else x)
    
    # Patch lỗi đường dẫn Windows nếu cần
    def patch_win(p):
        if isinstance(p, str) and 'data/raw/images/' in p:
            return 'data/raw/images/' + p.split('data/raw/images/')[-1]
        return p
    if not args.img_dir:
        df['image_path'] = df['image_path'].apply(patch_win)

    val_loader = DataLoader(MedicalImageTextDataset(df, transform, tokenizer), batch_size=16, shuffle=False)

    ckpt_input = args.checkpoint if args.checkpoint else config['training']['checkpoint_dir']
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_input, "*.pth"))) if os.path.isdir(ckpt_input) else [ckpt_input]

    results = []
    print(f"🚀 Đang đánh giá {len(ckpt_files)} checkpoints...")

    for ckpt_path in ckpt_files:
        filename = os.path.basename(ckpt_path)
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            
            model = MultimodalModel(config['model']['image_encoder'], config['model']['text_encoder']).to(device)
            state_dict = fix_state_dict(state_dict, model.state_dict().keys())
            model.load_state_dict(state_dict, strict=True)
            
            i2t, t2i = evaluate_retrieval(model, val_loader, device)
            results.append({'checkpoint': filename, 'R@1': f"{i2t[0]:.2f}%", 'R@5': f"{i2t[1]:.2f}%", 'R@10': f"{i2t[2]:.2f}%"})
            print(f"✅ {filename}: R@1 = {i2t[0]:.2f}%")
        except Exception as e:
            print(f"⚠️ Lỗi {filename}: {e}")

    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
