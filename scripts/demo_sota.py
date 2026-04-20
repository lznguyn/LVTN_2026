"""
demo_sota.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chạy thử NHANH với 1 ảnh + 1 báo cáo bất kỳ để kiểm tra xem
model SOTA (SwinV2-384) có trả về kết quả đúng hay không.

Cách dùng:
  python scripts/demo_sota.py --checkpoint <path_to_sota.pt>

Tuỳ chọn:
  --index   INT   Index của mẫu trong val.csv (mặc định: ngẫu nhiên)
  --topk    INT   Số kết quả trả về (mặc định: 5)
  --config  STR   Config YAML (mặc định: configs/default.yaml)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
import yaml
import random
from PIL import Image
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.models.projection import ProjectionHead
from scripts.evaluate import fix_state_dict, get_transforms
from scripts.test_sota import remap_sota_state_dict, patch_path


# ──────────────────────────────────────────────────────────────
# HÀM LOAD MODEL SOTA (tái sử dụng logic từ test_sota.py)
# ──────────────────────────────────────────────────────────────
def load_sota_model(checkpoint_path, config, device):
    img_enc_name  = "swinv2_base_window12to24_192to384"
    text_enc_name = config['model']['text_encoder']

    print(f"📦 Khởi tạo model: {img_enc_name}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Remap key nếu cần
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("image_encoder.model."):
        print("   🔄 Dịch key HuggingFace → timm ...")
        state_dict = remap_sota_state_dict(state_dict)

    # Tự động đọc dim từ checkpoint
    img_proj_w  = state_dict.get("image_proj.mlp.0.weight")
    txt_proj_w  = state_dict.get("text_proj.mlp.0.weight")
    last_proj_w = state_dict.get("image_proj.mlp.3.weight")

    img_input_dim  = img_proj_w.shape[1] if img_proj_w is not None else 1024
    img_hidden_dim = img_proj_w.shape[0] if img_proj_w is not None else 2048
    txt_input_dim  = txt_proj_w.shape[1] if txt_proj_w is not None else 768
    txt_hidden_dim = txt_proj_w.shape[0] if txt_proj_w is not None else 2048
    embed_dim      = last_proj_w.shape[0] if last_proj_w is not None else 512

    model = MultimodalModel(
        image_encoder_name=img_enc_name,
        text_model_name=text_enc_name,
        embed_dim=embed_dim
    ).to(device)
    model.image_proj = ProjectionHead(img_input_dim, embed_dim, img_hidden_dim).to(device)
    model.text_proj  = ProjectionHead(txt_input_dim, embed_dim, txt_hidden_dim).to(device)

    state_dict = fix_state_dict(state_dict, model.state_dict().keys())

    # Lọc key bị lỗi kích thước
    model_sd     = model.state_dict()
    filtered_sd  = {k: v for k, v in state_dict.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
    msg = model.load_state_dict(filtered_sd, strict=False)

    n_loaded  = len(filtered_sd)
    n_total   = len(model_sd)
    n_missing = len(msg.missing_keys)
    print(f"✅ Nạp xong: {n_loaded}/{n_total} keys | Missing: {n_missing}")
    if n_missing > n_total * 0.5:
        print("   ⚠️  CẢNH BÁO: >50% keys thiếu – kết quả KHÔNG có nghĩa!")

    model.eval()
    return model, img_enc_name


# ──────────────────────────────────────────────────────────────
# ENCODE TOÀN BỘ TẬP VAL  → gallery embeddings
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_gallery(model, val_df, image_transform, tokenizer, device, batch_size=32):
    from torch.utils.data import DataLoader
    from src.data.dataset import MedicalImageTextDataset

    dataset    = MedicalImageTextDataset(val_df, image_transform, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_img, all_txt = [], []
    print(f"🔄 Encoding gallery ({len(val_df)} mẫu) ...")
    from tqdm import tqdm
    for batch in tqdm(dataloader, leave=False, desc="Gallery"):
        img_e, txt_e = model(
            batch['image'].to(device),
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device)
        )
        all_img.append(img_e.cpu())
        all_txt.append(txt_e.cpu())

    return torch.cat(all_img), torch.cat(all_txt)


# ──────────────────────────────────────────────────────────────
# ENCODE 1 MẪU ĐƠN LẺ
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_single(model, image_path, report_text, image_transform, tokenizer, device, max_length=128):
    # Ảnh
    image  = Image.open(image_path).convert("RGB")
    img_t  = image_transform(image).unsqueeze(0).to(device)

    # Văn bản
    tok = tokenizer(
        report_text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    ids  = tok['input_ids'].to(device)
    mask = tok['attention_mask'].to(device)

    img_e, txt_e = model(img_t, ids, mask)
    return img_e.cpu(), txt_e.cpu()


# ──────────────────────────────────────────────────────────────
# IN KẾT QUẢ ĐẸP
# ──────────────────────────────────────────────────────────────
def print_results(query_idx, val_df, top_indices, similarities,
                  topk, mode="I2T"):
    """
    mode = "I2T" : dùng img_embed của query → tìm txt trong gallery
    mode = "T2I" : dùng txt_embed của query → tìm img trong gallery
    """
    row = val_df.iloc[query_idx]

    tag  = "🖼→📄 Image→Text" if mode == "I2T" else "📄→🖼 Text→Image"
    print(f"\n{'─'*62}")
    print(f"  {tag}  (Query Index = {query_idx})")
    print(f"{'─'*62}")
    print(f"  📁 Ảnh query : {os.path.basename(str(row['image_path']))}")
    txt_preview = str(row['report'])[:120].replace('\n', ' ')
    print(f"  📝 Báo cáo   : {txt_preview}...")

    if 'cluster_id' in val_df.columns:
        print(f"  🏷  Cluster   : {row['cluster_id']}")

    print(f"\n  ═══ TOP-{topk} KẾT QUẢ RETRIEVED ═══")
    hit_r1 = False
    for rank, (idx, sim) in enumerate(zip(top_indices, similarities), 1):
        matched      = "✅ ĐÚNG" if idx == query_idx else "❌"
        cluster_hint = ""
        if 'cluster_id' in val_df.columns:
            q_cl  = val_df.iloc[query_idx]['cluster_id']
            r_cl  = val_df.iloc[idx]['cluster_id']
            if idx != query_idx and q_cl == r_cl:
                cluster_hint = f"  [Cluster Match: {r_cl}]"
                matched      = "🟡 Cluster"
        retrieved_txt = str(val_df.iloc[idx]['report'])[:80].replace('\n', ' ')
        print(f"  #{rank:2d} [{sim:.4f}] {matched}{cluster_hint}")
        print(f"       → {retrieved_txt}...")
        if rank == 1 and idx == query_idx:
            hit_r1 = True

    print(f"\n  R@1 (Strict) cho mẫu này: {'✅ HIT' if hit_r1 else '❌ MISS'}")
    print(f"{'─'*62}\n")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Demo SOTA model với 1 mẫu bất kỳ")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Đường dẫn file .pt của SOTA model (mặc định: tự tìm trong /kaggle/input/my-sota-model/)")
    parser.add_argument('--config',     type=str, default='configs/default.yaml')
    parser.add_argument('--index',      type=int, default=None,
                        help="Index mẫu trong val.csv (mặc định: random)")
    parser.add_argument('--topk',       type=int, default=5,
                        help="Số kết quả top-K cần hiển thị (mặc định: 5)")
    parser.add_argument('--image_dir',  type=str, default=None)
    args = parser.parse_args()

    # ── Auto-detect checkpoint trên Kaggle nếu không truyền --checkpoint ──
    if args.checkpoint is None:
        # Các thư mục gốc cần tìm (Kaggle có 2 kiểu path)
        KAGGLE_SEARCH_ROOTS = [
            "/kaggle/input",                        # Dataset gắn thông thường
            "/kaggle/input/datasets",               # Dataset user-uploaded (nguyenabc/...)
        ]
        print("   🔍 Đang tìm file checkpoint (.pt/.pth) ...")
        found_pts = []
        for search_root in KAGGLE_SEARCH_ROOTS:
            if not os.path.exists(search_root):
                continue
            for root, dirs, files in os.walk(search_root):
                for fname in files:
                    if fname.endswith('.pt') or fname.endswith('.pth'):
                        full_path = os.path.join(root, fname)
                        if full_path not in found_pts:
                            found_pts.append(full_path)
        if found_pts:
            # Ưu tiên file lớn nhất (thường là model weights)
            found_pts.sort(key=lambda p: os.path.getsize(p), reverse=True)
            args.checkpoint = found_pts[0]
            print(f"   🤖 Auto-detect checkpoint: {args.checkpoint}")
            if len(found_pts) > 1:
                print(f"   ℹ️  Tìm thấy {len(found_pts)} files, dùng file lớn nhất.")
                for p in found_pts[1:3]:
                    print(f"      - {p} ({os.path.getsize(p)//1024//1024} MB)")
        else:
            print("   ⚠️  Không tìm thấy. Dùng --checkpoint <path> để chỉ rõ.")

    if args.checkpoint is None:
        parser.error("Chưa tìm thấy checkpoint. Dùng --checkpoint <path> để chỉ rõ.")

    print("=" * 62)
    print("🚀  DEMO SOTA MODEL — Kiểm Tra Nhanh 1 Mẫu")
    print("=" * 62)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️  Device: {device}")

    # Load config
    with open(args.config, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)

    # Load val_df & fix path
    val_df = pd.read_csv(config['data']['val_csv'])
    print(f"📊 Val set: {len(val_df)} mẫu")

    image_dir = args.image_dir

    # ── Auto-detect trên Kaggle ───────────────────────────────
    if image_dir is None and os.path.exists("/kaggle/input"):
        # Ưu tiên 1: thư mục chuẩn của Indiana University dataset
        IU_FIXED = "/kaggle/input/chest-xrays-indiana-university/images/images_normalized"
        if os.path.isdir(IU_FIXED):
            image_dir = IU_FIXED
            print(f"   📁 Auto-detect Kaggle image dir: {IU_FIXED}")
        else:
            # Ưu tiên 2: tự tìm file ảnh đầu tiên trong val_df
            first_img   = val_df['image_path'].iloc[0]
            first_fname = os.path.basename(str(first_img).replace('\\', '/'))
            print(f"   🔍 Tìm '{first_fname}' trong /kaggle/input...")
            for root, _, files in os.walk("/kaggle/input"):
                if first_fname in files:
                    image_dir = root
                    print(f"   ✅ Tìm thấy tại: {image_dir}")
                    break
        if image_dir is None:
            print("   ⚠️  Không tìm thấy thư mục ảnh. Dùng --image_dir để chỉ rõ.")
    # ──────────────────────────────────────────────────────────

    val_df['image_path'] = val_df['image_path'].apply(
        lambda p: patch_path(p, image_dir)
    )

    # Chọn query index
    query_idx = args.index if args.index is not None else random.randint(0, len(val_df) - 1)
    print(f"🎯 Query index: {query_idx} (dùng --index N để chọn cụ thể)")

    # Chuẩn bị tokenizer & transforms
    text_enc_name   = config['model']['text_encoder']
    tokenizer       = AutoTokenizer.from_pretrained(text_enc_name)
    image_transform = get_transforms(384)   # SOTA dùng 384

    # Load model
    model, _ = load_sota_model(args.checkpoint, config, device)

    # Encode gallery (toàn bộ val set)
    gallery_img_emb, gallery_txt_emb = encode_gallery(
        model, val_df, image_transform, tokenizer, device
    )

    # Encode query
    q_row     = val_df.iloc[query_idx]
    q_img_emb, q_txt_emb = encode_single(
        model,
        str(q_row['image_path']),
        str(q_row['report']),
        image_transform,
        tokenizer,
        device
    )

    topk = min(args.topk, len(val_df))

    # ── I2T: Query bằng Ảnh → tìm Báo cáo phù hợp ──
    sim_i2t  = torch.matmul(q_img_emb, gallery_txt_emb.t()).squeeze(0)
    top_i2t  = torch.topk(sim_i2t, topk)
    print_results(query_idx, val_df,
                  top_i2t.indices.tolist(),
                  top_i2t.values.tolist(),
                  topk, mode="I2T")

    # ── T2I: Query bằng Báo cáo → tìm Ảnh phù hợp ──
    sim_t2i  = torch.matmul(q_txt_emb, gallery_img_emb.t()).squeeze(0)
    top_t2i  = torch.topk(sim_t2i, topk)
    print_results(query_idx, val_df,
                  top_t2i.indices.tolist(),
                  top_t2i.values.tolist(),
                  topk, mode="T2I")

    # ── Tóm tắt ──
    hit_i2t = (top_i2t.indices[0].item() == query_idx)
    hit_t2i = (top_t2i.indices[0].item() == query_idx)

    print("=" * 62)
    print("📋  TÓM TẮT")
    print(f"   Image→Text R@1 : {'✅ HIT' if hit_i2t else '❌ MISS'}")
    print(f"   Text→Image R@1 : {'✅ HIT' if hit_t2i else '❌ MISS'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
