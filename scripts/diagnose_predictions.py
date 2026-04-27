import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import os
import sys
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint .pth file")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--num_cases', type=int, default=10, help="Number of failed cases to print")
    args = parser.parse_args()

    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    image_size = config['data']['image_size']
    if "384" in args.checkpoint:
        image_size = 384
        
    image_transform = get_transforms(image_size)
    val_df = pd.read_csv(config['data']['val_csv'])
    
    def patch_path(p):
        if not isinstance(p, str): return p
        p = p.replace('\\', '/')
        if 'data/raw/images/' in p:
            return 'data/raw/images/' + p.split('data/raw/images/')[-1]
        return p
    
    val_df['image_path'] = val_df['image_path'].apply(patch_path)

    # Dùng 500 mẫu đầu tiên để chạy cho nhanh
    subset_df = val_df.head(500).reset_index(drop=True)
    val_loader = DataLoader(MedicalImageTextDataset(subset_df, image_transform, tokenizer), batch_size=32, shuffle=False)

    img_enc_name = config['model']['image_encoder']
    if "384" in args.checkpoint:
        img_enc_name = "swinv2_base_window12_384"
        
    model = MultimodalModel(
        img_enc_name, 
        config['model']['text_encoder'],
        embed_dim=config['model'].get('embed_dim', 512)
    ).to(device)
    
    print(f"Loading checkpoint {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    state_dict = fix_state_dict(state_dict, model.state_dict().keys())
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_img_embeds = []
    all_txt_embeds = []
    all_clusters = []
    all_reports = []

    print("Extracting embeddings for diagnosis...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Retrieval Embeddings", leave=False):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img_embeds, txt_embeds, _ = model(images, input_ids, attention_mask)
            
            all_img_embeds.append(img_embeds.cpu())
            all_txt_embeds.append(txt_embeds.cpu())
            all_reports.extend(batch['raw_report'])
            
            if 'cluster_id' in batch:
                all_clusters.append(batch['cluster_id'].cpu())

    img_embeds = torch.cat(all_img_embeds, dim=0)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)
    clusters = torch.cat(all_clusters, dim=0) if all_clusters else None
    
    print("Computing similarities and finding errors...")
    sim_matrix = torch.matmul(img_embeds.to(device), txt_embeds.to(device).t())
    
    # Lấy top 1 prediction
    top1_indices = torch.argmax(sim_matrix, dim=1).cpu()
    
    num_printed = 0
    print(f"\n{'='*80}")
    print(f"🔥 TOP {args.num_cases} FAILED PREDICTIONS (DIAGNOSIS) 🔥")
    print(f"{'='*80}")
    
    for i in range(len(top1_indices)):
        pred_idx = top1_indices[i].item()
        target_idx = i
        
        # Kiểm tra xem có đúng chính xác bệnh nhân không
        is_exact_match = (pred_idx == target_idx)
        
        # Kiểm tra xem có cùng cluster không (nếu có cluster)
        is_cluster_match = False
        if clusters is not None:
            c_query = clusters[i].item()
            c_pred = clusters[pred_idx].item()
            if c_query != -1 and c_query == c_pred:
                is_cluster_match = True
                
        # Nếu mô hình đoán sai hoàn toàn (không exact match và không cluster match)
        if not is_exact_match and not is_cluster_match:
            print(f"\n❌ [CASE {num_printed + 1}]")
            print(f"🖼️ Ảnh ID / Index: {i} | Image Path: {subset_df.iloc[i]['image_path']}")
            print(f"🟩 Ground Truth (Cluster {clusters[i].item() if clusters is not None else 'N/A'}):")
            print(f"   {all_reports[i]}")
            print(f"🟥 Top-1 Predicted (Cluster {clusters[pred_idx].item() if clusters is not None else 'N/A'}):")
            print(f"   {all_reports[pred_idx]}")
            print("-" * 80)
            
            num_printed += 1
            if num_printed >= args.num_cases:
                break
                
    if num_printed == 0:
        print("Tất cả mẫu đều dự đoán đúng hoặc nằm trong cùng Cluster!")

if __name__ == "__main__":
    main()
