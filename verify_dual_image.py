import torch
from src.models.hrgr_agent import HRGRAgent
import sys
import yaml

def test_hrgr_multi_image():
    # Load config to get standard parameters
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Mock parameters
    vocab_size = 1000
    templates = ["Normal chest.", "Cardiac silhouette is normal."]
    # We will use the image encoder from config
    image_encoder = config['model']['image_encoder']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Init Model
    try:
        model = HRGRAgent(
            image_encoder_name=image_encoder,
            vocab_size=vocab_size,
            templates=templates
        ).to(device)
        print("[OK] Model Initialization Successful!")
    except Exception as e:
        print(f"[FAIL] Model Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Mock inputs (B, C, H, W)
    batch_size = 2
    img_new = torch.randn(batch_size, 3, 256, 256).to(device)
    img_old = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print("\n--- Testing Forward Pass ---")
    try:
        p, s, w = model(img_new, img_old)
        print(f"Policy Logits Shape: {p.shape}") # Expect (2, 8, num_templates + 1)
        print(f"Stop Logits Shape: {s.shape}")   # Expect (2, 8, 1)
        print(f"Word Logits Shape: {w.shape}")   # Expect (2, 8, 20, 1000)
        print("[OK] Forward Pass Successful!")
    except Exception as e:
        print(f"[FAIL] Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Generation ---")
    class MockVocab:
        def __init__(self):
            self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: 'normal', 4: 'lungs', 5: 'clear'}
    
    try:
        # Test generate with single image (auto-fallback in model)
        res1 = model.generate(img_new[0:1], vocab=MockVocab())
        print(f"AI Report (Single): {res1}")
        
        # Test generate with dual image
        res2 = model.generate(img_new[0:1], img_old[0:1], vocab=MockVocab())
        print(f"AI Report (Dual): {res2}")
        print("[OK] Generation Successful!")
    except Exception as e:
        print(f"[FAIL] Generation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hrgr_multi_image()
