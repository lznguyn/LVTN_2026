import torch
import torchvision.transforms as transforms
from PIL import Image
from src.models.hrgr_agent import HRGRAgent
from src.data.vocabulary import WordVocabulary
import json
import yaml
import sys
import os

def generate_report(image_path, model_path):
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

    # 5. Generate Report
    print(f"\n--- Generating Report for: {os.path.basename(image_path)} ---")
    report = model.generate(image_tensor, vocab)
    print(f"AI REPORT: {report}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_report.py <image_path> [model_checkpoint]")
        # Placeholder for demo if no args provided
        test_img = "data/raw/sample.jpg" # Update with a real path if testing
        if os.path.exists(test_img):
            generate_report(test_img, "checkpoints/hrgr_epoch_1.pth")
    else:
        img_p = sys.argv[1]
        ckpt_p = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/hrgr_epoch_1.pth"
        generate_report(img_p, ckpt_p)
