# src/segmentation/train_segmentation.py
# FedMedSeg Phase 2 — Full Training Loop
#
# TRAINING STRATEGY:
#   Phase A (Epochs 1-10):   Encoder FROZEN, only Decoder trains.
#                             This lets the Decoder stabilize first.
#   Phase B (Epochs 11-end): Unfreeze last 5 Encoder blocks for fine-tuning.
#                             Lower learning rate to avoid catastrophic forgetting.
#
# OPTIMIZER:   Adam  (Adaptive Moment Estimation)
#   Formula: θ = θ - (η / (√v̂ + ε)) * m̂
#   lr=1e-4, weight_decay=1e-5
#
# SCHEDULER:  ReduceLROnPlateau
#   Reduces learning rate by ×0.5 if val_dice doesn't improve for 5 epochs.
#
# EARLY STOPPING:
#   Stops training if val_dice doesn't improve for 10 epochs.
#
# CHECKPOINTING:
#   Saves model whenever a new best val_dice is achieved.
#
# All metrics are logged to:  results/segmentation/training_logs.csv
# Publication-ready plots:    results/segmentation/loss_curves.pdf
#                             results/segmentation/dice_iou_curves.pdf

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
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for servers/notebooks
import matplotlib.pyplot as plt
import numpy as np
import argparse

# ── Path setup so imports work from project root ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet    import MobileNetV2UNet
from segmentation.loss          import DiceBCELoss
from segmentation.metrics       import compute_all_metrics
from segmentation.dataset_rsna  import RSNAPneumoniaDataset
from segmentation.device_utils  import add_device_arg, get_device


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these values before training
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Paths
    "rsna_root":      str(PROJECT_ROOT / "data" / "rsna_pneumonia"),
    "train_csv":      str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "train_subset.csv"),
    "val_csv":        str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "val_subset.csv"),
    "results_dir":    str(PROJECT_ROOT / "results" / "segmentation"),
    "checkpoint_dir": str(PROJECT_ROOT / "results" / "segmentation"),

    # Training Hyperparameters
    "num_epochs":        30,       # Total epochs (Phase A + Phase B)
    "phase_b_start":     10,       # Epoch at which we unfreeze encoder
    "batch_size":        16,       # Images per mini-batch
    "num_workers":       4,        # DataLoader parallel workers
    "lr_phase_a":        1e-4,     # Learning rate — frozen encoder phase
    "lr_phase_b":        1e-5,     # Lower LR —  fine-tuning phase
    "weight_decay":      1e-5,     # L2 regularization

    # Scheduler
    "scheduler_factor":   0.5,     # Multiply LR by this on plateau
    "scheduler_patience": 5,       # Epochs with no improvement before LR drop

    # Early Stopping
    "early_stop_patience": 10,     # Epochs with no improvement before stopping

    # Model
    "pretrained":          True,   # Load ImageNet weights for MobileNetV2
    "prediction_threshold":0.5,    # Probability threshold for binary mask

    # Reproducibility
    "random_seed": 42,
}
# ═══════════════════════════════════════════════════════════════════════════════


# ImageNet normalization stats — required because MobileNetV2 encoder
# was pre-trained on ImageNet with these exact values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(augment: bool):
    """Return image transforms for training (with augmentation) or validation."""
    if augment:
        return T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2),  # Only on image, not mask
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Training one epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """
    Run one full pass over the training data.

    Returns:
        dict: {"train_loss": float, "train_dice": float, "train_iou": float,
               "train_pixel_acc": float}
    """
    model.train()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        # Forward pass
        preds = model(images)   # preds are probabilities (after sigmoid in model)

        # Compute hybrid Dice-BCE loss
        # Note: DiceBCELoss expects raw logits, but our model returns sigmoid output.
        # We pass preds directly and DiceLoss handles it correctly since values are in [0,1].
        # BCEWithLogitsLoss requires logits, so we use BCELoss here instead.
        loss = criterion(preds, masks)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics (no gradient needed)
        with torch.no_grad():
            m = compute_all_metrics(preds, masks, threshold=CONFIG["prediction_threshold"])

        total_loss += loss.item()
        all_dice.append(m["dice"])
        all_iou.append(m["iou"])
        all_pix.append(m["pixel_acc"])

        # Progress log every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch [{epoch}/{total_epochs}] "
                  f"Step [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}  Dice: {m['dice']:.4f}")

    return {
        "train_loss":      total_loss / len(loader),
        "train_dice":      np.mean(all_dice),
        "train_iou":       np.mean(all_iou),
        "train_pixel_acc": np.mean(all_pix),
    }


# ── Validation one epoch ──────────────────────────────────────────────────────

def validate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set.

    Returns:
        dict: {"val_loss": float, "val_dice": float, "val_iou": float,
               "val_pixel_acc": float}
    """
    model.eval()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)

            m = compute_all_metrics(preds, masks, threshold=CONFIG["prediction_threshold"])

            total_loss += loss.item()
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_pix.append(m["pixel_acc"])

    return {
        "val_loss":      total_loss / len(loader),
        "val_dice":      np.mean(all_dice),
        "val_iou":       np.mean(all_iou),
        "val_pixel_acc": np.mean(all_pix),
    }


# ── Plot generation (publication-ready) ──────────────────────────────────────

def save_training_plots(log_path: Path, results_dir: Path):
    """
    Read training_logs.csv and save two publication-ready PDFs:
      1. loss_curves.pdf   — Train vs Validation Loss
      2. dice_iou_curves.pdf — Train/Val Dice and IoU

    Uses a clean, academic plot style suitable for papers.
    """
    import pandas as pd

    df = pd.read_csv(log_path)
    epochs = df["epoch"].values

    # ── Style ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":    "DejaVu Serif",
        "font.size":      12,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
    })

    # ── Plot 1: Loss Curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, df["train_loss"], label="Training Loss",   color="#2196F3", linewidth=2)
    ax.plot(epochs, df["val_loss"],   label="Validation Loss", color="#F44336",
            linewidth=2, linestyle="--")
    ax.set_title("Dice-BCE Loss During Training", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curves.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(results_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Dice & IoU Curves ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, df["train_dice"], label="Train Dice", color="#4CAF50", linewidth=2)
    ax1.plot(epochs, df["val_dice"],   label="Val Dice",   color="#FF9800",
             linewidth=2, linestyle="--")
    ax1.set_title("Dice Coefficient", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Dice Score")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(epochs, df["train_iou"], label="Train IoU", color="#9C27B0", linewidth=2)
    ax2.plot(epochs, df["val_iou"],   label="Val IoU",   color="#009688",
             linewidth=2, linestyle="--")
    ax2.set_title("Mean IoU (Jaccard Index)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU Score")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle("Segmentation Metrics — MobileNetV2-UNet", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "dice_iou_curves.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(results_dir / "dice_iou_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved loss_curves.pdf and dice_iou_curves.pdf")


# ── Sample Prediction Visualization ──────────────────────────────────────────

def save_prediction_samples(model, val_loader, device, results_dir: Path, n_samples: int = 20):
    """
    Save side-by-side comparison images for the first `n_samples` validation images.
    Columns: [Original X-ray | Ground Truth Mask | Predicted Mask | Overlay]
    """
    model.eval()
    samples_dir  = results_dir / "prediction_samples"
    overlay_dir  = results_dir / "prediction_overlay"
    samples_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    count = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if count >= n_samples:
                break

            images_gpu = images.to(device)
            preds      = model(images_gpu).cpu()

            for i in range(images.size(0)):
                if count >= n_samples:
                    break

                # Denormalize image for display
                img  = images[i].numpy().transpose(1, 2, 0)  # (H, W, 3)
                img  = (img * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
                img  = np.clip(img, 0, 1)

                gt_mask   = masks[i, 0].numpy()                  # (H, W)
                pred_mask = (preds[i, 0].numpy() > 0.5).astype(float) # (H, W)

                # Overlay: X-ray in grayscale, predicted mask in red
                overlay = img.copy()
                overlay[:, :, 0] = np.where(pred_mask == 1, 1.0, overlay[:, :, 0])
                overlay[:, :, 1] = np.where(pred_mask == 1, 0.0, overlay[:, :, 1])
                overlay[:, :, 2] = np.where(pred_mask == 1, 0.0, overlay[:, :, 2])

                # 4-panel figure
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img);              axes[0].set_title("X-ray");             axes[0].axis("off")
                axes[1].imshow(gt_mask, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                axes[2].imshow(pred_mask, cmap="gray"); axes[2].set_title("Prediction");  axes[2].axis("off")
                axes[3].imshow(overlay);          axes[3].set_title("Overlay");           axes[3].axis("off")
                fig.suptitle(f"Sample {count + 1}", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(samples_dir / f"sample_{count+1:03d}.png", dpi=120, bbox_inches="tight")
                plt.close()

                # Overlay-only image for hero figures
                import matplotlib
                matplotlib.image.imsave(str(overlay_dir / f"overlay_{count+1:03d}.png"), overlay)

                count += 1

    print(f"  ✓ Saved {count} prediction samples to {samples_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  FedMedSeg Phase 2 — Segmentation Training")
    print("=" * 65)

    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="FedMedSeg Segmentation Training")
    add_device_arg(parser)
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = get_device(args.device)

    # ── Set random seeds for reproducibility ──────────────────────────────────
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    # ── Results directory ─────────────────────────────────────────────────────
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(CONFIG["checkpoint_dir"]) / "best_model_weights.pth"

    # Save config for reproducibility
    CONFIG["timestamp"]  = datetime.now().isoformat()
    CONFIG["device"]     = str(device)
    with open(results_dir / "training_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = RSNAPneumoniaDataset(
        rsna_root      = CONFIG["rsna_root"],
        subset_csv     = CONFIG["train_csv"],
        img_transform  = get_transforms(augment=True),
        augment        = True,
    )
    val_dataset = RSNAPneumoniaDataset(
        rsna_root      = CONFIG["rsna_root"],
        subset_csv     = CONFIG["val_csv"],
        img_transform  = get_transforms(augment=False),
        augment        = False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,
        num_workers = CONFIG["num_workers"],
        pin_memory  = device.type in ("cuda", "mps"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        pin_memory  = device.type in ("cuda", "mps"),
    )
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding MobileNetV2-UNet...")
    model = MobileNetV2UNet(
        pretrained     = CONFIG["pretrained"],
        freeze_encoder = True,     # Phase A: Encoder frozen
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — Total: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = DiceBCELoss(smooth=1e-6)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CONFIG["lr_phase_a"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "max",   # We want to MAXIMIZE val_dice
        factor    = CONFIG["scheduler_factor"],
        patience  = CONFIG["scheduler_patience"],
    )

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_path = results_dir / "training_logs.csv"
    csv_headers = [
        "epoch", "phase",
        "train_loss", "val_loss",
        "train_dice", "val_dice",
        "train_iou",  "val_iou",
        "train_pixel_acc", "val_pixel_acc",
        "learning_rate", "epoch_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_dice      = 0.0
    early_stop_counter = 0
    phase              = "A"

    print("\n--- Starting Training ---\n")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start = time.time()

        # ── Phase B: Unfreeze encoder at phase_b_start ────────────────────────
        if epoch == CONFIG["phase_b_start"] and phase == "A":
            phase = "B"
            print(f"\n{'='*55}")
            print(f"  PHASE B: Unfreezing last 5 Encoder blocks (epoch {epoch})")
            print(f"{'='*55}\n")
            model.unfreeze_encoder(num_blocks=5)

            # Reset optimizer with lower learning rate for fine-tuning
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr           = CONFIG["lr_phase_b"],
                weight_decay = CONFIG["weight_decay"],
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max",
                factor=CONFIG["scheduler_factor"],
                patience=CONFIG["scheduler_patience"],
            )

        # ── Train & Validate ──────────────────────────────────────────────────
        print(f"\n[Epoch {epoch}/{CONFIG['num_epochs']}]  Phase {phase}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, CONFIG["num_epochs"])
        val_metrics   = validate(model, val_loader, criterion, device)

        # Learning rate step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["val_dice"])
        epoch_time = time.time() - epoch_start

        # ── Print epoch summary ───────────────────────────────────────────────
        print(f"\n  ┌──────────────────────────────────────────────────────┐")
        print(f"  │ Epoch {epoch:2d} Summary (Phase {phase})                        │")
        print(f"  ├────────────────────────┬────────────┬────────────────┤")
        print(f"  │ Metric                 │   Train    │   Validation   │")
        print(f"  ├────────────────────────┼────────────┼────────────────┤")
        print(f"  │ Loss (Dice-BCE)        │ {train_metrics['train_loss']:.4f}     │ {val_metrics['val_loss']:.4f}          │")
        print(f"  │ Dice Coefficient ↑     │ {train_metrics['train_dice']:.4f}     │ {val_metrics['val_dice']:.4f}          │")
        print(f"  │ Mean IoU ↑             │ {train_metrics['train_iou']:.4f}     │ {val_metrics['val_iou']:.4f}          │")
        print(f"  │ Pixel Accuracy ↑       │ {train_metrics['train_pixel_acc']:.4f}     │ {val_metrics['val_pixel_acc']:.4f}          │")
        print(f"  └────────────────────────┴────────────┴────────────────┘")
        print(f"  LR: {current_lr:.2e}  |  Time: {epoch_time:.1f}s")

        # ── Log to CSV ────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase,
                f"{train_metrics['train_loss']:.6f}", f"{val_metrics['val_loss']:.6f}",
                f"{train_metrics['train_dice']:.6f}", f"{val_metrics['val_dice']:.6f}",
                f"{train_metrics['train_iou']:.6f}",  f"{val_metrics['val_iou']:.6f}",
                f"{train_metrics['train_pixel_acc']:.6f}", f"{val_metrics['val_pixel_acc']:.6f}",
                f"{current_lr:.2e}",  f"{epoch_time:.2f}",
            ])

        # ── Checkpoint: Save best model ───────────────────────────────────────
        if val_metrics["val_dice"] > best_val_dice:
            best_val_dice      = val_metrics["val_dice"]
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ NEW BEST — val_dice = {best_val_dice:.4f}  →  Model saved!")
        else:
            early_stop_counter += 1
            print(f"  No improvement ({early_stop_counter}/{CONFIG['early_stop_patience']})")

        # ── Early Stopping ────────────────────────────────────────────────────
        if early_stop_counter >= CONFIG["early_stop_patience"]:
            print(f"\n  ⚠  Early stopping triggered at epoch {epoch}.")
            print(f"  Best val_dice: {best_val_dice:.4f}")
            break

    # ── Post-training ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"  Best Validation Dice: {best_val_dice:.4f}")
    print(f"  Weights saved to:     {checkpoint_path}")
    print(f"{'='*55}\n")

    # Generate plots
    print("Generating training plots...")
    save_training_plots(log_path, results_dir)

    # Load best model and generate prediction samples
    print("\nGenerating prediction samples with best model...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    save_prediction_samples(model, val_loader, device, results_dir, n_samples=20)

    # Save final evaluation report
    report = {
        "training_complete": True,
        "best_val_dice":     best_val_dice,
        "total_epochs_run":  epoch,
        "timestamp":         datetime.now().isoformat(),
        "checkpoint_path":   str(checkpoint_path),
        "config":            CONFIG,
    }
    with open(results_dir / "model_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ All results saved to: {results_dir}/")
    print("  ├── training_logs.csv")
    print("  ├── loss_curves.pdf + .png")
    print("  ├── dice_iou_curves.pdf + .png")
    print("  ├── training_config.json")
    print("  ├── model_evaluation_report.json")
    print("  ├── best_model_weights.pth")
    print("  ├── prediction_samples/  (20 PNG comparisons)")
    print("  └── prediction_overlay/  (colored overlays)")


if __name__ == "__main__":
    main()
