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

        # --- TỰ ĐỘNG ĐỌC KÍCH THƯỚC TỪ CHECKPOINT ---
        # Tránh lỗi size mismatch do SOTA dùng hidden_dim khác
        img_proj_w = state_dict.get("image_proj.mlp.0.weight")
        txt_proj_w = state_dict.get("text_proj.mlp.0.weight")
        if img_proj_w is not None:
            img_hidden_dim = img_proj_w.shape[0]   # [hidden, input]
            img_input_dim  = img_proj_w.shape[1]
            print(f"   📐 Tự động nhận diện Image Proj: input={img_input_dim}, hidden={img_hidden_dim}")
        else:
            img_hidden_dim = 2048
            img_input_dim  = 1024
        
        if txt_proj_w is not None:
            txt_hidden_dim = txt_proj_w.shape[0]
            txt_input_dim  = txt_proj_w.shape[1]
            print(f"   📐 Tự động nhận diện Text  Proj: input={txt_input_dim}, hidden={txt_hidden_dim}")
        else:
            txt_hidden_dim = 2048

        # Determine embed_dim from the last layer of image_proj
        last_proj_w = state_dict.get("image_proj.mlp.3.weight")
        embed_dim = last_proj_w.shape[0] if last_proj_w is not None else 512
        print(f"   📐 Embed dim = {embed_dim}")

        # Build model với đúng embed_dim từ checkpoint
        from src.models.projection import ProjectionHead
        model = MultimodalModel(image_encoder_name=img_enc_name,
                                text_model_name=text_enc_name,
                                embed_dim=embed_dim).to(device)
        # Override projection heads với đúng hidden_dim
        model.image_proj = ProjectionHead(img_input_dim, embed_dim, img_hidden_dim).to(device)
        model.text_proj  = ProjectionHead(txt_input_dim, embed_dim, txt_hidden_dim).to(device)
        # ------------------------------------------------

        state_dict = fix_state_dict(state_dict, model.state_dict().keys())
        
        # --- LỌC BỎ KEY BỊ LỖII KÍCH THƯỚC TRƯỚC KHI LOAD ---
        model_sd = model.state_dict()
        filtered_sd = {}
        skipped_size = []
        for k, v in state_dict.items():
            if k in model_sd:
                if v.shape == model_sd[k].shape:
                    filtered_sd[k] = v
                else:
                    skipped_size.append(f"{k}: checkpoint{list(v.shape)} vs model{list(model_sd[k].shape)}")
            # key không tồn tại trong model -> để strict=False xử lý
        
        if skipped_size:
            print(f"\n   ⚠️  Bỏ qua {len(skipped_size)} keys bị lỗi kích thước:")
            for s in skipped_size[:5]:
                print(f"      - {s}")
            if len(skipped_size) > 5:
                print(f"      ... và {len(skipped_size)-5} keys khác")
        # -------------------------------------------------------

        msg = model.load_state_dict(filtered_sd, strict=False)
        
        n_loaded = len(filtered_sd)
        n_total = len(model_sd)
        n_missing = len(msg.missing_keys)
        print(f"\n✨ Nạp model xong! Loaded: {n_loaded}/{n_total} keys | Missing: {n_missing}")
        
        if n_missing > n_total * 0.5:
            print(f"   🚨 CẢNH BÁO NGHIÊM TRỌNG: >50% keys bị thiếu!")
            print(f"   File SOTA được train với kiến trúc KHÁC HOÀN TOÀN với code hiện tại.")
            print(f"   Kết quả evaluate sẽ KHÔNG CÓ Ý NGHĨA (model đang dùng weights ngẫu nhiên).")
        elif n_missing > 30:
            print(f"   ⚠️  Cảnh báo: {n_missing} keys bị thiếu, sẽ dùng weights mặc định (random init).")
    except Exception as e:
        import traceback
        print(f"❌ Lỗi khi nạp weights: {e}")
        traceback.print_exc()
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
