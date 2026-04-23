import os
import yaml
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multimodal import MultimodalModel
from src.data.dataset import MedicalImageTextDataset
from src.engine.trainer import MultimodalTrainer
from scripts.evaluate import evaluate_retrieval

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)

def get_train_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomHorizontalFlip(), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    print("==================================================================")
    print("LUAN VAN: HUAN LUYEN MULTIMODAL SOFT CLUSTERING-GUIDED NEGATIVE SAMPLING")
    print("==================================================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    config = load_config()
    
    print("\n[1] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    train_transform = get_train_transforms(config['data']['image_size'])
    val_transform = get_val_transforms(config['data']['image_size'])
    
    print("[2] Loading Database...")
    try:
        train_df = pd.read_csv(config['data']['train_csv'])
        val_df = pd.read_csv(config['data']['val_csv'])
        
        processed_dir = "data/processed"
        train_soft_path = os.path.join(processed_dir, "soft_labels_train.npy")
        val_soft_path = os.path.join(processed_dir, "soft_labels_val.npy")
        
        train_soft, val_soft = None, None
        if os.path.exists(train_soft_path) and os.path.exists(val_soft_path):
            train_soft = np.load(train_soft_path)
            val_soft   = np.load(val_soft_path)
            print(f"Loaded soft labels: Train {train_soft.shape} | Val {val_soft.shape}")
        else:
            print("Warning: soft_labels (.npy) not found. Run scripts/create_clusters.py first.")
        
        ext_csv = config['data'].get('ext_eval_csv', "")
        if ext_csv and os.path.exists(ext_csv):
            print(f"Loading external evaluation dataset (LMOD): {ext_csv}")
            full_df = pd.read_csv(ext_csv)
            full_soft = None 
            img_prefix = config['data'].get('ext_eval_img_dir', "")
            if img_prefix:
                full_df['image_path'] = full_df['image_path'].apply(lambda x: os.path.join(img_prefix, x) if not os.path.isabs(x) else x)
            print(f"Full Dataset (External): {len(full_df)} samples")
        else:
            full_df   = pd.concat([train_df, val_df], ignore_index=True)
            full_soft = None
            if train_soft is not None and val_soft is not None:
                full_soft = np.concatenate([train_soft, val_soft], axis=0)
            print(f"Full Dataset (IU-Xray): {len(full_df)} samples")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_dataset = MedicalImageTextDataset(train_df, train_transform, tokenizer, config['data']['max_text_length'], soft_labels=train_soft)
    val_dataset   = MedicalImageTextDataset(val_df,   val_transform,   tokenizer, config['data']['max_text_length'], soft_labels=val_soft)
    full_dataset  = MedicalImageTextDataset(full_df,  val_transform,   tokenizer, config['data']['max_text_length'], soft_labels=full_soft)
    
    num_workers = 0 if os.name == 'nt' else config['training']['num_workers']
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
    full_loader  = DataLoader(full_dataset,  batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
    
    print("\n[3] Building Model...")
    model = MultimodalModel(config['model']['image_encoder'], config['model']['text_encoder'], config['model']['embed_dim'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print("\n[4] Starting Training...")
    trainer = MultimodalTrainer(model, config, device=device)
    
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, "training_history.csv")
    history = []
    
    start_epoch = 1
    best_r1 = -1.0
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if os.path.exists(last_ckpt_path):
        checkpoint = trainer.load_checkpoint(last_ckpt_path)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_r1 = checkpoint.get('best_r1', -1.0)
            if os.path.exists(history_path):
                try: history = pd.read_csv(history_path).to_dict('records')
                except: pass

    eval_every = config['training'].get('eval_every_n_epochs', 1)

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)
        
        current_r1 = 0.0
        if epoch % eval_every == 0:
            try:
                print(f"Evaluating Validation...")
                r_strict_val, r_cluster_val = evaluate_retrieval(trainer.model, val_loader, device)
                
                label_full = "LMOD" if ext_csv and os.path.exists(ext_csv) else "Full Data"
                print(f"Evaluating {label_full}...")
                r_strict_full, r_cluster_full = evaluate_retrieval(trainer.model, full_loader, device)
                
                history.append({
                    'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                    'r1_val': r_cluster_val[0], 'r1_full': r_cluster_full[0]
                })
                pd.DataFrame(history).to_csv(history_path, index=False)
                current_r1 = r_cluster_full[0]
                print(f"Epoch {epoch} - R@1 [{label_full}]: {current_r1:.2f}%")
            except Exception as e:
                print(f"Evaluation error: {e}")
        
        if current_r1 > best_r1:
            best_r1 = current_r1
            model_module = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
            torch.save(model_module.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

        trainer.save_checkpoint(last_ckpt_path, epoch, best_r1)
        model_module = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
        torch.save(model_module.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))

    print("\n DONE!")

if __name__ == "__main__":
    main()
