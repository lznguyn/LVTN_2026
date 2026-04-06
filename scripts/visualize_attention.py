import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
import yaml
import sys
import re
from src.models.hrgr_agent import HRGRAgent
from src.data.vocabulary import WordVocabulary

# Global list to capture alphas via hooks
captured_alphas = []

def attention_hook_fn(module, input, output):
    # output: (attention_weighted_encoding, alpha)
    # alpha: (Batch, NumPixels)
    captured_alphas.append(output[1].detach().cpu())

def visualize_attention_non_invasive(image_path, checkpoint_path, output_dir="results_viz"):
    # Clear previous captures
    captured_alphas.clear()
    
    # 1. Load Config & Resources
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("data/processed/templates.json", "r", encoding="utf-8") as f:
        templates = json.load(f)
    vocab = WordVocabulary.load("data/processed/vocab.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Initialize Model
    model = HRGRAgent(
        image_encoder_name=config['model']['image_encoder'],
        vocab_size=len(vocab),
        templates=templates
    )
    
    # Identify Epoch from path for naming
    epoch_match = re.search(r'epoch_(\d+)', checkpoint_path)
    epoch_num = epoch_match.group(1) if epoch_match else "unknown"
    
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Register Hook to capture Attention
    # We hook into model.attention module
    hook_handle = model.attention.register_forward_hook(attention_hook_fn)

    # 4. Preprocess Image
    img_size = config['data']['image_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 5. Generate Report (This will trigger the hooks)
    report = model.generate(image_tensor, vocab)
    hook_handle.remove() # Clean up hook
    
    # Re-extract sentences to match alphas
    sentences = [s.strip() for s in re.split(r'[.;?!\n]', report) if s.strip()]
    
    print(f"Report: {report}")
    print(f"Captured {len(captured_alphas)} attention maps.")

    # 6. Visualization using Matplotlib
    os.makedirs(output_dir, exist_ok=True)
    num_viz = min(len(sentences), len(captured_alphas))
    
    if num_viz == 0:
        print("Cảnh báo: Không trích xuất được Attention hoặc Sentences.")
        return

    fig, axes = plt.subplots(num_viz + 1, 1, figsize=(10, 5 * (num_viz + 1)))
    if num_viz == 0: axes = [axes] # Handle single plot case
    
    # Row 0: Original
    axes[0].imshow(raw_image)
    axes[0].set_title(f"X-Ray - Epoch {epoch_num}", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Grid size detection (usually 12x12 or 16x16 for SwinV2)
    # Total pixels = captured_alphas[0].shape[1]
    total_pixels = captured_alphas[0].shape[1]
    side = int(np.sqrt(total_pixels))
    
    for i in range(num_viz):
        text = sentences[i]
        alpha = captured_alphas[i].reshape(side, side).numpy()
        
        # Resize and normalize
        alpha_resized = cv2.resize(alpha, (raw_image.width, raw_image.height))
        alpha_norm = (alpha_resized - alpha_resized.min()) / (alpha_resized.max() - alpha_resized.min() + 1e-8)
        
        # Heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * alpha_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np.array(raw_image), 0.6, heatmap, 0.4, 0)
        
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"Sentence {i+1}: {text}", fontsize=12, wrap=True)
        axes[i+1].axis('off')

    plt.tight_layout()
    output_filename = f"viz_epoch_{epoch_num}_{os.path.basename(image_path)}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualization saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/visualize_attention.py <image_path> <checkpoint_path>")
    else:
        visualize_attention_non_invasive(sys.argv[1], sys.argv[2])
