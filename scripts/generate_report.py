import torch
import torchvision.transforms as transforms
from PIL import Image
from src.models.hrgr_agent import HRGRAgent
from src.data.vocabulary import WordVocabulary
import json
import yaml
import sys
import os

def generate_report(image_path, old_image_path=None, model_path="checkpoints/hrgr_epoch_1.pth"):
    # 1. Load Config & Resources
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
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
    
    # 3. Load Trained Weights
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model path {model_path} not found. Running with initial weights.")
    
    model.to(device)
    model.eval()

    # 4. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    old_image_tensor = None
    if old_image_path:
        old_image = Image.open(old_image_path).convert('RGB')
        old_image_tensor = transform(old_image).unsqueeze(0).to(device)

    # 5. Generate Report
    print(f"\n--- Generating Report ---")
    print(f"New Image: {os.path.basename(image_path)}")
    if old_image_path:
        print(f"Old Image: {os.path.basename(old_image_path)}")
    else:
        print(f"Old Image: (None - using new image as baseline)")
        
    report = model.generate(image_tensor, old_image_tensor, vocab)
    print(f"\nAI REPORT: {report}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate medical report from one or two images.")
    parser.add_argument("image", help="Path to the new/current image.")
    parser.add_argument("--old", help="Path to the old/baseline image (optional).", default=None)
    parser.add_argument("--ckpt", help="Path to the model checkpoint.", default="checkpoints/hrgr_epoch_1.pth")
    
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        generate_report(args.image, args.old, args.ckpt)
    else:
        print(f"Error: Image {args.image} not found.")
