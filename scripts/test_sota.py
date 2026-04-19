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

def remap_sota_state_dict(state_dict):
    """
    Dịch key của file SOTA (HuggingFace style) sang cấu trúc MultimodalModel (timm style).
    
    Mapping chính:
      image_encoder.embeddings.*         -> image_encoder.model.patch_embed.*
      image_encoder.encoder.layers.*     -> image_encoder.model.layers.*
      image_encoder.layernorm.*          -> image_encoder.model.norm.*
      text_encoder.embeddings.*          -> text_encoder.model.embeddings.*
      text_encoder.encoder.*             -> text_encoder.model.encoder.*
      text_encoder.pooler.*              -> text_encoder.model.pooler.*
      img_proj.proj.*                    -> image_proj.mlp.*
      txt_proj.proj.*                    -> text_proj.mlp.*
    """
    new_sd = {}
    skipped = []
    for k, v in state_dict.items():
        nk = k

        # === IMAGE ENCODER ===
        if nk.startswith("image_encoder.embeddings.patch_embeddings.projection."):
            nk = nk.replace("image_encoder.embeddings.patch_embeddings.projection.",
                             "image_encoder.model.patch_embed.proj.")
        elif nk.startswith("image_encoder.embeddings.norm."):
            nk = nk.replace("image_encoder.embeddings.norm.",
                             "image_encoder.model.patch_embed.norm.")
        elif nk.startswith("image_encoder.layernorm."):
            nk = nk.replace("image_encoder.layernorm.",
                             "image_encoder.model.norm.")
        elif nk.startswith("image_encoder.encoder.layers."):
            # Phức tạp: dịch từ HF attention style -> timm attn style
            nk = nk.replace("image_encoder.encoder.layers.", "image_encoder.model.layers.")
            nk = nk.replace(".attention.self.logit_scale", ".attn.logit_scale")
            nk = nk.replace(".attention.self.continuous_position_bias_mlp.", ".attn.cpb_mlp.")
            nk = nk.replace(".attention.self.query.", ".attn.q_bias_")   # placeholder
            nk = nk.replace(".attention.self.key.", ".attn.qkv_k_")      # placeholder
            nk = nk.replace(".attention.self.value.", ".attn.v_bias_")   # placeholder
            nk = nk.replace(".attention.output.dense.", ".attn.proj.")
            nk = nk.replace(".layernorm_before.", ".norm1.")
            nk = nk.replace(".layernorm_after.", ".norm2.")
            nk = nk.replace(".intermediate.dense.", ".mlp.fc1.")
            nk = nk.replace(".output.dense.", ".mlp.fc2.")
            nk = nk.replace(".downsample.", ".downsample.")

        # === TEXT ENCODER ===
        elif nk.startswith("text_encoder.embeddings."):
            nk = nk.replace("text_encoder.embeddings.", "text_encoder.model.embeddings.")
        elif nk.startswith("text_encoder.encoder."):
            nk = nk.replace("text_encoder.encoder.", "text_encoder.model.encoder.")
        elif nk.startswith("text_encoder.pooler."):
            nk = nk.replace("text_encoder.pooler.", "text_encoder.model.pooler.")

        # === PROJECTION HEADS ===
        elif nk.startswith("img_proj.proj."):
            nk = nk.replace("img_proj.proj.", "image_proj.mlp.")
        elif nk.startswith("txt_proj.proj."):
            nk = nk.replace("txt_proj.proj.", "text_proj.mlp.")

        # Bỏ qua logit_scale (không có trong MultimodalModel)
        elif nk == "logit_scale":
            skipped.append(k)
            continue

        new_sd[nk] = v

    if skipped:
        print(f"   ℹ️  Bỏ qua {len(skipped)} keys không tương thích: {skipped[:3]}...")
    return new_sd


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
        
        # Tự động nhận diện và dịch key từ kiến trúc SOTA -> MultimodalModel
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith("image_encoder.model."):
            print("   🔄 Đang dịch key từ HuggingFace format -> timm format...")
            state_dict = remap_sota_state_dict(state_dict)
        
        state_dict = fix_state_dict(state_dict, model.state_dict().keys())
        msg = model.load_state_dict(state_dict, strict=False)
        
        n_missing = len(msg.missing_keys)
        n_unexpected = len(msg.unexpected_keys)
        print(f"✨ Nạp model xong! Missing: {n_missing} | Unexpected: {n_unexpected}")
        if n_missing > 50:
            print(f"   ⚠️  Cảnh báo: Quá nhiều key bị thiếu ({n_missing}). "
                  f"File SOTA có thể được train với kiến trúc hoàn toàn khác.")
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
