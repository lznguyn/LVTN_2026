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
    """
    Sửa lỗi mapping tên giữa các bản timm và xử lý thay đổi kiến trúc (thêm Dropout).
    """
    has_layers_dot = any("layers." in k for k in model_keys)
    has_layers_underscore = any("layers_" in k for k in model_keys)
    
    # Kiểm tra xem model hiện tại có dùng cấu trúc MLP mới (có Dropout nên layer index bị đẩy lên) hay không
    is_new_model = any(".mlp.5." in k for k in model_keys)
    is_old_checkpoint = is_new_model and not any(".mlp.5." in k for k in state_dict.keys())
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        
        # 1. Sửa lỗi timm (layers. vs layers_)
        if has_layers_dot and "layers_" in k:
            new_k = new_k.replace("layers_", "layers.")
        elif has_layers_underscore and "layers." in k:
            new_k = new_k.replace("layers.", "layers_")
            
        # 2. Xử lý ánh xạ lại lớp Projection nếu nạp model cũ vào kiến trúc mới (có Dropout)
        if is_old_checkpoint:
            if ".mlp.4." in new_k:
                new_k = new_k.replace(".mlp.4.", ".mlp.5.")
            elif ".mlp.3." in new_k:
                new_k = new_k.replace(".mlp.3.", ".mlp.4.")
                
        new_state_dict[new_k] = v
    return new_state_dict

def get_ground_truth_action(report, templates):
    """
    Tìm Action chuẩn (0: Generate, 1..N: Template) cho câu đầu tiên của báo cáo.
    Dùng logic tương tự HRGRRLTrainer.
    """
    sentences = re.split(r'[.;?!\n]', str(report))
    # Lấy câu đầu tiên có nội dung
    first_sentence = ""
    for s in sentences:
        s = s.strip()
        if s:
            first_sentence = s
            break
    
    if not first_sentence:
        return 0
        
    for idx, t in enumerate(templates):
        if first_sentence.lower() == t.lower():
            return idx + 1 # Action index
    return 0 # Generate

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, return_embeds=False):
    """Đánh giá R@K cho MultimodalModel (Stage 1) - Hỗ trợ Semantic Cluster Match"""
    model.eval()
    all_img_embeds = []
    all_txt_embeds = []
    all_clusters = []

    for batch in tqdm(dataloader, desc="Retrieval Embeddings", leave=False):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        img_embeds, txt_embeds, _ = model(images, input_ids, attention_mask)
        
        all_img_embeds.append(img_embeds.cpu())
        all_txt_embeds.append(txt_embeds.cpu())
        
        if 'cluster_id' in batch:
            all_clusters.append(batch['cluster_id'].cpu())

    img_embeds = torch.cat(all_img_embeds, dim=0)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)
    
    # Gộp tất cả các cluster của dataset lại
    clusters = torch.cat(all_clusters, dim=0) if all_clusters else None
    
    # Nếu chỉ cần lấy embeddings để vẽ biểu đồ (t-SNE)
    if return_embeds:
        return img_embeds, txt_embeds, clusters

    # [1] Tính R@K theo chuẩn Strict (phải khớp chính xác Patient ID)
    r_strict = calculate_recall_chunked(img_embeds, txt_embeds, device, clusters=None)
    
    # [2] Tính R@K theo chuẩn Cluster-Aware (khớp Patient ID hoặc cùng Cluster)
    r_cluster = calculate_recall_chunked(img_embeds, txt_embeds, device, clusters=clusters) if clusters is not None else r_strict
    
    return r_strict, r_cluster

@torch.no_grad()
def evaluate_agent_accuracy(model, dataloader, device, templates):
    """Đánh giá chính xác Action chọn Template của Agent và độ đa dạng dự đoán"""
    model.eval()
    total_samples = 0
    correct_actions = 0
    
    # Theo dõi xem model dự đoán ra bao nhiêu loại template khác nhau
    pred_actions_all = []
    
    for batch in tqdm(dataloader, desc="Agent Accuracy", leave=False):
        images = batch['image'].to(device)
        reports = batch['raw_report']
        
        # Forward lấy prediction (chỉ lấy câu đầu tiên)
        policy_logits, _, _ = model(images)
        preds = policy_logits[:, 0, :].argmax(dim=-1) # (B,)
        pred_actions_all.extend(preds.cpu().tolist())
        
        # Tính Ground Truth Action cho từng mẫu trong batch
        targets = []
        for r in reports:
            targets.append(get_ground_truth_action(r, templates))
        targets = torch.tensor(targets, device=device)
        
        correct_actions += (preds == targets).sum().item()
        total_samples += images.size(0)

    acc = (correct_actions / total_samples) * 100 if total_samples > 0 else 0
    
    # Tính độ đa dạng (Diversity): Số lượng Action khác nhau được dự đoán
    unique_preds = len(set(pred_actions_all))
    
    return (acc, unique_preds, 0), (acc, unique_preds, 0)

def calculate_recall_chunked(query_embeds, gallery_embeds, device, clusters=None, chunk_size=1000):
    """
    Tính R@K theo chiến lược Clustering-Guided (Task 3 - Luận văn):

    Một kết quả truy xuất được tính là ĐÚNG nếu:
      (a) Exact Match:   Đúng index bệnh nhân gốc (i == j)
      (b) Cluster Match: Cùng cluster_id → cùng nhóm bệnh ngữ nghĩa
                         (khác bệnh nhân nhưng cùng "loại bệnh" → không phải Negative thật sự)

    Điều này phù hợp với Task 3: "Nếu ảnh A và báo cáo B thuộc cùng một cụm,
    sẽ KHÔNG coi chúng là mẫu âm tính của nhau".
    """
    num_queries = query_embeds.size(0)
    hits_r1, hits_r5, hits_r10 = 0, 0, 0
    cluster_match_count = 0  # Đếm số lần cluster match được kích hoạt

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
            # ─── Task 3: Clustering-Guided Recall ───────────────────────────
            query_clusters = clusters[start:end].unsqueeze(1)      # (chunk, 1)
            retrieved_clusters = clusters[top10_indices]           # (chunk, K)

            # (a) Khớp chính xác index bệnh nhân
            matches_exact = (top10_indices == targets)

            # (b) Khớp ngữ nghĩa: cùng nhóm bệnh theo cluster
            is_valid_cluster = (query_clusters != -1)              # Bỏ nhãn lỗi -1
            matches_cluster = (retrieved_clusters == query_clusters) & is_valid_cluster

            # Gộp: đúng nếu exact HOẶC semantic cluster match
            matches = matches_exact | matches_cluster
            cluster_match_count += (matches_cluster[:, :1].any(dim=1) &
                                    ~matches_exact[:, :1].any(dim=1)).sum().item()
            # ────────────────────────────────────────────────────────────────
        else:
            matches = (top10_indices == targets)

        hits_r1  += matches[:, :1].any(dim=1).sum().item()
        hits_r5  += matches[:, :5].any(dim=1).sum().item()
        hits_r10 += matches[:, :10].any(dim=1).sum().item()

    r1  = (hits_r1  / num_queries) * 100
    r5  = (hits_r5  / num_queries) * 100
    r10 = (hits_r10 / num_queries) * 100

    if clusters is not None and cluster_match_count > 0:
        print(f"       📌 Cluster-match cứu thêm {cluster_match_count} mẫu vào R@1 "
              f"({cluster_match_count/num_queries*100:.1f}% của tổng)")

    return r1, r5, r10

def main():
    parser = argparse.ArgumentParser(description='Smart Evaluate Retrieval & Agent')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output', type=str, default='data/evaluation_summary.csv')
    parser.add_argument('--image_size', type=int, help='Override image size (e.g. 384)')
    parser.add_argument('--image_encoder', type=str, help='Override image encoder name (e.g. swinv2_base_window12_384)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Resources
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    # [CUSTOM] Tự động nhận diện size 384 từ tên file nếu không truyền tham số
    image_size = args.image_size
    if image_size is None and args.checkpoint and "384" in args.checkpoint:
        image_size = 384
        print(f"✨ Tự động nhận diện Image Size = {image_size} từ tên file.")
    
    image_size = image_size if image_size else config['data']['image_size']
    image_transform = get_transforms(image_size)
    val_df = pd.read_csv(config['data']['val_csv'])
    
    # --- VÁ LỖI ĐƯỜNG DẪN WINDOWS TUYỆT ĐỐI ---
    def patch_path(p):
        if not isinstance(p, str): return p
        p = p.replace('\\', '/')
        if 'data/raw/images/' in p:
            return 'data/raw/images/' + p.split('data/raw/images/')[-1]
        return p
    
    print("🛠️  Đang sửa lỗi đường dẫn Windows -> Linux...")
    val_df['image_path'] = val_df['image_path'].apply(patch_path)
    if 'old_image_path' in val_df.columns:
        val_df['old_image_path'] = val_df['old_image_path'].apply(patch_path)
    # ------------------------------------------

    val_loader = DataLoader(MedicalImageTextDataset(val_df, image_transform, tokenizer), batch_size=16, shuffle=False)

    with open("data/processed/templates.json", "r", encoding="utf8") as f:
        templates = json.load(f)
    with open("data/processed/vocab.json", "r", encoding="utf8") as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data['word2idx'])

    ckpt_input = args.checkpoint if args.checkpoint else config['training']['checkpoint_dir']
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_input, "*.pth"))) if os.path.isdir(ckpt_input) else [ckpt_input]

    results = []
    model_retrieval = None
    model_agent = None

    print(f"\n🚀 Đang đánh giá {len(ckpt_files)} file checkpoints...")

    for ckpt_path in ckpt_files:
        filename = os.path.basename(ckpt_path)
        print(f"\n➔ Đang xử lý: {filename}")
        
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            is_agent = any('sentence_decoder' in k for k in state_dict.keys())
            
            if is_agent:
                if model_agent is None:
                    model_agent = HRGRAgent(config['model']['image_encoder'], vocab_size, templates).to(device)
                
                # Fix state dict dựa trên keys của model hiện tại
                state_dict = fix_state_dict(state_dict, model_agent.state_dict().keys())
                msg = model_agent.load_state_dict(state_dict, strict=False)
                
                print(f"   📊 Load status: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys.")
                if len(msg.missing_keys) > 100:
                    print("   ⚠️  Cảnh báo: Có quá nhiều keys bị thiếu, khả năng cao là nạp model thất bại!")
                
                i2t, t2i = evaluate_agent_accuracy(model_agent, val_loader, device, templates)
                type_str = "Agent (Acc)"
            else:
                if model_retrieval is None:
                    # [CUSTOM] Ghi đè bộ mã hóa ảnh nếu cần (ví dụ bản SOTA dùng base-384)
                    img_enc_name = args.image_encoder if args.image_encoder else config['model']['image_encoder']
                    if args.image_encoder is None and "384" in filename:
                        img_enc_name = "swinv2_base_window12_384"
                        print(f"   ✨ Tự động gán Image Encoder = {img_enc_name}")
                        
                    model_retrieval = MultimodalModel(img_enc_name, config['model']['text_encoder']).to(device)
                
                state_dict = fix_state_dict(state_dict, model_retrieval.state_dict().keys())
                model_retrieval.load_state_dict(state_dict, strict=True)
                model_retrieval.eval()
                # Đồng bộ DataLoader với Image Size mới
                val_loader_fixed = DataLoader(MedicalImageTextDataset(val_df, image_transform, tokenizer), batch_size=16, shuffle=False)
                i2t, t2i = evaluate_retrieval(model_retrieval, val_loader_fixed, device)
                type_str = "Retrieval"

            # Lưu kết quả linh hoạt theo loại model
            res_entry = {
                'checkpoint': filename, 
                'type': type_str,
            }
            
            if is_agent:
                res_entry['R@1 (Acc)'] = f"{i2t[0]:.2f}%"
                res_entry['Diversity'] = f"{i2t[1]}"
                res_entry['R@5'] = "-"
            else:
                res_entry['R@1 (Acc)'] = f"{i2t[0]:.2f}%"
                res_entry['Diversity'] = "-"
                res_entry['R@5'] = f"{i2t[1]:.2f}%"
                
            results.append(res_entry)
            print(f"✅ {type_str} - Main Metric: {i2t[0]:.2f}%")
            
        except Exception as e:
            print(f"⚠️ Lỗi: {str(e)}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print("\n" + "="*50)
        print(df.to_string(index=False))
        print("="*50)

if __name__ == "__main__":
    main()
