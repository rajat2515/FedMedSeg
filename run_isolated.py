"""
run_isolated.py
===============
FedMedSeg Phase 3 — Isolated Training Experiment (The Control Group)

PURPOSE:
  Trains two INDEPENDENT MobileNetV2-UNet models, one per hospital client,
  each using ONLY its own Non-IID data. No collaboration, no weight sharing.

  This experiment proves WHY federated learning is necessary:
    - Client A (75% pneumonia) → over-predicts pneumonia → high False Positives
    - Client B (75% normal)    → under-predicts pneumonia → high False Negatives
    - Both models are WORSE than the centralized baseline

  Without this experiment, we cannot scientifically justify federation.

RUN:
    cd /home/rajat/Documents/Project/FedMedSeg
    .venv/bin/python run_isolated.py

OPTIONAL FLAGS:
    --device cpu|cuda|mps   (default: auto-detect)
    --epochs 15             (default: 15)
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Third-Party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project Imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet
from segmentation.loss import DiceBCELoss
from segmentation.metrics import compute_all_metrics
from segmentation.dataset_rsna import RSNAPneumoniaDataset


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONFIG = {
    "rsna_root":      str(PROJECT_ROOT / "data" / "rsna_pneumonia"),
    "client_a_csv":   str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "client_a_train.csv"),
    "client_b_csv":   str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "client_b_train.csv"),
    "val_csv":        str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "val_subset.csv"),
    "results_dir":    str(PROJECT_ROOT / "results" / "federated" / "isolated"),

    "max_epochs":     15,
    "lr":             1e-4,
    "weight_decay":   1e-5,
    "grad_clip":      1.0,
    "batch_size":     16,
    "num_workers":    0,
    "threshold":      0.5,
    "random_seed":    42,
}


def get_train_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    elif preference == "mps" or (
        preference == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN & VALIDATE — Reused for both clients
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
        optimizer.step()

        with torch.no_grad():
            m = compute_all_metrics(preds, masks, threshold=CONFIG["threshold"])

        total_loss += loss.item()
        all_dice.append(m["dice"])
        all_iou.append(m["iou"])
        all_pix.append(m["pixel_acc"])

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{m['dice']:.4f}"})

    return {
        "train_loss":      total_loss / len(loader),
        "train_dice":      float(np.mean(all_dice)),
        "train_iou":       float(np.mean(all_iou)),
        "train_pixel_acc": float(np.mean(all_pix)),
    }


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Validation", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)
            m     = compute_all_metrics(preds, masks, threshold=CONFIG["threshold"])

            total_loss += loss.item()
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_pix.append(m["pixel_acc"])

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{m['dice']:.4f}"})

    return {
        "val_loss":      total_loss / len(loader),
        "val_dice":      float(np.mean(all_dice)),
        "val_iou":       float(np.mean(all_iou)),
        "val_pixel_acc": float(np.mean(all_pix)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE CLIENT — Full isolated training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_client(
    client_name: str,
    train_csv: str,
    val_csv: str,
    results_dir: Path,
    device: torch.device,
    max_epochs: int,
):
    """Train a single client model in isolation."""

    print(f"\n{'='*60}")
    print(f"  Training: {client_name}")
    print(f"{'='*60}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = RSNAPneumoniaDataset(
        rsna_root=CONFIG["rsna_root"],
        subset_csv=train_csv,
        img_transform=get_train_transform(),
        augment=True,
    )
    val_ds = RSNAPneumoniaDataset(
        rsna_root=CONFIG["rsna_root"],
        subset_csv=val_csv,
        img_transform=get_val_transform(),
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MobileNetV2UNet(pretrained=True, freeze_encoder=False).to(device)
    criterion = DiceBCELoss(smooth=1e-6)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ── CSV Logger ────────────────────────────────────────────────────────────
    log_path = results_dir / f"training_logs_{client_name.lower().replace(' ', '_')}.csv"
    csv_headers = [
        "epoch", "train_loss", "val_loss",
        "train_dice", "val_dice",
        "train_iou", "val_iou",
        "train_pixel_acc", "val_pixel_acc",
        "epoch_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_headers)

    best_val_dice = 0.0
    best_model_path = results_dir / f"{client_name.lower().replace(' ', '_')}_model.pth"

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m   = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        print(f"  [{client_name}] Epoch {epoch:>2}/{max_epochs}  "
              f"Train Dice: {train_m['train_dice']:.4f}  "
              f"Val Dice: {val_m['val_dice']:.4f}  "
              f"({epoch_time:.1f}s)")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_m['train_loss']:.6f}", f"{val_m['val_loss']:.6f}",
                f"{train_m['train_dice']:.6f}", f"{val_m['val_dice']:.6f}",
                f"{train_m['train_iou']:.6f}",  f"{val_m['val_iou']:.6f}",
                f"{train_m['train_pixel_acc']:.6f}", f"{val_m['val_pixel_acc']:.6f}",
                f"{epoch_time:.2f}",
            ])

        # Save best
        if val_m["val_dice"] > best_val_dice:
            best_val_dice = val_m["val_dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ NEW BEST — val_dice = {best_val_dice:.4f}")

    # ── Final Evaluation ──────────────────────────────────────────────────────
    print(f"\n  [{client_name}] Loading best model (val_dice={best_val_dice:.4f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    final_m = validate(model, val_loader, criterion, device)

    result = {
        "client_name":   client_name,
        "best_val_dice": best_val_dice,
        "final_metrics": final_m,
        "epochs":        max_epochs,
        "train_samples": len(train_ds),
    }

    print(f"\n  [{client_name}] Final Results:")
    print(f"    Dice:       {final_m['val_dice']:.4f}")
    print(f"    IoU:        {final_m['val_iou']:.4f}")
    print(f"    Pixel Acc:  {final_m['val_pixel_acc']:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Isolated Training Experiment")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device = get_device(args.device)
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = args.epochs

    print("\n" + "=" * 65)
    print("  ISOLATED TRAINING EXPERIMENT — The Control Group")
    print("  Why? Proves that training alone on biased data FAILS.")
    print("  This motivates the need for Federated Learning.")
    print("=" * 65)

    # ── Train Client A ────────────────────────────────────────────────────────
    result_a = train_client(
        client_name="Client_A",
        train_csv=CONFIG["client_a_csv"],
        val_csv=CONFIG["val_csv"],
        results_dir=results_dir,
        device=device,
        max_epochs=max_epochs,
    )

    # ── Train Client B ────────────────────────────────────────────────────────
    result_b = train_client(
        client_name="Client_B",
        train_csv=CONFIG["client_b_csv"],
        val_csv=CONFIG["val_csv"],
        results_dir=results_dir,
        device=device,
        max_epochs=max_epochs,
    )

    # ── Save combined report ──────────────────────────────────────────────────
    report = {
        "experiment":    "Isolated Training (No Federation)",
        "purpose":       "Control group — proves biased local data leads to poor models",
        "completed_at":  datetime.now().isoformat(),
        "device":        str(device),
        "config":        CONFIG,
        "client_a":      result_a,
        "client_b":      result_b,
    }

    report_path = results_dir / "isolated_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ISOLATED TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"\n  {'Model':<20}  {'Val Dice':<12}  {'Val IoU':<12}  {'Val PixAcc'}")
    print(f"  {'─'*60}")
    print(f"  {'Client A (75% Pneu)':<20}  "
          f"{result_a['final_metrics']['val_dice']:.4f}        "
          f"{result_a['final_metrics']['val_iou']:.4f}        "
          f"{result_a['final_metrics']['val_pixel_acc']:.4f}")
    print(f"  {'Client B (75% Norm)':<20}  "
          f"{result_b['final_metrics']['val_dice']:.4f}        "
          f"{result_b['final_metrics']['val_iou']:.4f}        "
          f"{result_b['final_metrics']['val_pixel_acc']:.4f}")
    print(f"  {'Centralized (ref)':<20}  0.6234        0.5609        0.9516")
    print(f"\n  → Both isolated models should be WORSE than centralized.")
    print(f"  → This proves the need for Federated Learning!\n")
    print(f"  Results saved to: {results_dir}/")
    print(f"    ├── client_a_model.pth")
    print(f"    ├── client_b_model.pth")
    print(f"    ├── training_logs_client_a.csv")
    print(f"    ├── training_logs_client_b.csv")
    print(f"    └── isolated_metrics.json")


if __name__ == "__main__":
    main()
