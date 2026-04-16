# src/segmentation/train_ablation.py
import os
import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

# ── Path setup so imports work from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet    import MobileNetV2UNet, CustomCNNUnet
from segmentation.loss          import DiceBCELoss
from segmentation.metrics       import compute_all_metrics
from segmentation.dataset_rsna  import RSNAPneumoniaDataset
from segmentation.train_segmentation import get_transforms, train_one_epoch, validate
from segmentation.device_utils import add_device_arg, get_device, safe_empty_cache

CONFIG = {
    "rsna_root":      str(PROJECT_ROOT / "data" / "rsna_pneumonia"),
    "train_csv":      str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "train_subset.csv"),
    "val_csv":        str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "val_subset.csv"),
    "results_dir":    str(PROJECT_ROOT / "results" / "ablation"),
    "num_epochs":     12,      # Short run for trajectory comparison
    "batch_size":     16,
    "num_workers":    0,       # 0 on Windows to avoid multiprocessing deadlock
    "lr_init":        1e-4,
    "random_seed":    42,
}

MODELS_CONFIG = {
    "Model 1 (2-Block)": {
        "class": "CustomCNNUnet", "kwargs": {"depth": 2}, "unfreeze_at": None
    },
    "Model 2 (3-Block)": {
        "class": "CustomCNNUnet", "kwargs": {"depth": 3}, "unfreeze_at": None
    },
    "Model 3A (Feature Extract)": {
        "class": "MobileNetV2UNet", "kwargs": {"pretrained": True, "freeze_encoder": True}, "unfreeze_at": None
    },
    "Model 3B (Partial Tune)": {
        "class": "MobileNetV2UNet", "kwargs": {"pretrained": True, "freeze_encoder": True}, "unfreeze_at": 5
    },
    "Model 3C (Full Fine-Tune)": {
        "class": "MobileNetV2UNet", "kwargs": {"pretrained": True, "freeze_encoder": False}, "unfreeze_at": None
    }
}

def build_model(model_conf: dict):
    if model_conf["class"] == "CustomCNNUnet":
        return CustomCNNUnet(**model_conf["kwargs"])
    elif model_conf["class"] == "MobileNetV2UNet":
        return MobileNetV2UNet(**model_conf["kwargs"])
    else:
        raise ValueError("Unknown model class")

def train_model_configuration(model_name: str, model_conf: dict, train_loader, val_loader, device, results_dir):
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}\n")
    
    model = build_model(model_conf).to(device)
    criterion = DiceBCELoss(smooth=1e-6)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr_init"])
    
    history = {"epoch": [], "val_dice": [], "train_dice": [], "val_iou": []}
    
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # Handle unfreezing for Model 3B
        if model_conf["unfreeze_at"] == epoch:
            print(f"   [!] Unfreezing top encoder layers for fine-tuning...")
            model.unfreeze_encoder(num_blocks=5)
            # Re-init optimizer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
            
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, CONFIG["num_epochs"])
        val_metrics   = validate(model, val_loader, criterion, device)
        
        print(f"   => Epoch {epoch}: Val Dice = {val_metrics['val_dice']:.4f} | Train Dice = {train_metrics['train_dice']:.4f}")
        
        history["epoch"].append(epoch)
        history["val_dice"].append(val_metrics["val_dice"])
        history["train_dice"].append(train_metrics["train_dice"])
        history["val_iou"].append(val_metrics["val_iou"])
        
    # Free memory
    del model
    safe_empty_cache(device)
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Run 5-Model Ablation Study")
    parser.add_argument("--model", type=str, choices=list(MODELS_CONFIG.keys()), 
                        help="Specify which model to run. If omitted, runs all 5 sequentially.")
    add_device_arg(parser)
    args = parser.parse_args()

    print("\nStarting 5-Model Ablation Study\n")
    device = get_device(args.device)
    
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = RSNAPneumoniaDataset(CONFIG["rsna_root"], CONFIG["train_csv"], get_transforms(augment=True), True)
    val_dataset   = RSNAPneumoniaDataset(CONFIG["rsna_root"], CONFIG["val_csv"], get_transforms(augment=False), False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    models_to_run = {args.model: MODELS_CONFIG[args.model]} if args.model else MODELS_CONFIG
    all_histories = {}
    
    for model_name, model_conf in models_to_run.items():
        hist = train_model_configuration(model_name, model_conf, train_loader, val_loader, device, results_dir)
        all_histories[model_name] = hist
        
        # Save individual CSV out immediately so distributed machines can save their piece
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        df_ind = pd.DataFrame(hist)
        df_ind.to_csv(results_dir / f"ablation_{safe_name}.csv", index=False)
        print(f"  ✓ Saved individual log for {model_name}")

    if not args.model:    
        # Generate Joint Plots ONLY if running all together
        print("\nGenerating final comparison plots...")
        
        plt.figure(figsize=(10, 6))
        for name, hist in all_histories.items():
            plt.plot(hist["epoch"], hist["val_dice"], marker='o', linewidth=2, label=name)
            
        plt.title("Val. Dice Coefficient Trajectory (Higher is better)", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Dice Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "ablation_val_dice.png", dpi=150)
        plt.close()
        
        flat_data = {"epoch": list(range(1, CONFIG["num_epochs"] + 1))}
        for name, hist in all_histories.items():
            flat_data[f"{name}_val_dice"] = hist["val_dice"]
            flat_data[f"{name}_val_iou"] = hist["val_iou"]
            
        df = pd.DataFrame(flat_data)
        df.to_csv(results_dir / "ablation_results_master.csv", index=False)
    
    print(f"\n✓ Study complete! Results saved in {results_dir}")

if __name__ == "__main__":
    main()
