import torch
import pandas as pd
from torch.utils.data import DataLoader
import os
import sys
import argparse
import yaml
from transformers import AutoTokenizer

# Cấu hình đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.data.dataset import MedicalImageTextDataset
from scripts.evaluate import evaluate_retrieval, fix_state_dict, get_transforms

def patch_path(p):
    if not isinstance(p, str): return p
    p = p.replace('\\', '/')
    if 'data/raw/images/' in p:
        return 'data/raw/images/' + p.split('data/raw/images/')[-1]
    return p

def main():
    print("==========================================================")
    print("🚀 SCRIPT ĐÁNH GIÁ ĐỘC LẬP CHO MÔ HÌNH SOTA (SwinV2 384)")
    print("==========================================================")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Đường dẫn tới file .pt")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(args.config, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)

    # --- CẤU HÌNH CỨNG CHO MÔ HÌNH SOTA ---
    img_enc_name = "swinv2_base_window12to24_192to384"
    image_size = 384
    text_enc_name = config['model']['text_encoder']
    # -------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(text_enc_name)
    image_transform = get_transforms(image_size)
    val_df = pd.read_csv(config['data']['val_csv'])
    
    print("🛠️ Đang sửa lỗi đường dẫn Windows -> Linux (nếu có)...")
    val_df['image_path'] = val_df['image_path'].apply(patch_path)

    val_loader = DataLoader(MedicalImageTextDataset(val_df, image_transform, tokenizer), batch_size=16, shuffle=False)

    print(f"\n📦 Đang khởi tạo mô hình với Image Size = {image_size} & Encoder = {img_enc_name}...")
    model = MultimodalModel(image_encoder_name=img_enc_name, text_model_name=text_enc_name).to(device)

    print(f"➔ Đang nạp weights từ: {args.checkpoint}")
    try:
        state_dict = torch.load(args.checkpoint, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        state_dict = fix_state_dict(state_dict, model.state_dict().keys())
        msg = model.load_state_dict(state_dict, strict=True)
        print("✨ Nạp model thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi nạp weights: {e}")
        return

    print("\n🔍 Bắt đầu chạy test case...")
    r_strict, r_cluster = evaluate_retrieval(model, val_loader, device)

    print("\n" + "="*50)
    print("🏆 KẾT QUẢ TEST SOTA MODEL")
    print("="*50)
    print(f"► Kết quả Khớp Chính xác (Strict):")
    print(f"   R@1:  {r_strict[0]:.2f}%")
    print(f"   R@5:  {r_strict[1]:.2f}%")
    print(f"   R@10: {r_strict[2]:.2f}%")
    print("-" * 50)
    print(f"► Kết quả Khớp Ngữ nghĩa Bệnh (Cluster):")
    print(f"   R@1:  {r_cluster[0]:.2f}%")
    print(f"   R@5:  {r_cluster[1]:.2f}%")
    print(f"   R@10: {r_cluster[2]:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
