"""
train_model3c_final.py
======================
FedMedSeg — Final Production Training for Model 3C
(MobileNetV2-UNet, Full Fine-Tune, Max 100 Epochs)

ANTI-OVERFITTING SAFEGUARDS ACTIVE:
  1.  Heavy Augmentation     — random flip, rotation, brightness, contrast,
                                Gaussian noise, and random erasing on EVERY batch
  2.  Two-Phase Training     — Phase A (frozen encoder, LR=1e-4) for the first
                                15 epochs, then Phase B (full fine-tune, LR=1e-5)
  3.  ReduceLROnPlateau      — halves LR if val_dice stalls for 7 epochs
  4.  Cosine Annealing       — smooth LR decay within each phase
  5.  Gradient Clipping      — prevents exploding gradients (||grad|| <= 1.0)
  6.  L2 Weight Decay        — 1e-5 mild regularization on all parameters
  7.  Early Stopping         — halts if val_dice doesn't improve for 15 epochs
  8.  Checkpointing          — saves weights ONLY on a NEW best val_dice

Run:
    cd /home/rajat/Documents/Project/FedMedSeg
    .venv/bin/python train_model3c_final.py

Optional flags:
    --device cpu|cuda|mps   (default: auto-detect)
    --resume                (resume from last checkpoint if it exists)
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
matplotlib.use("Agg")   # Non-interactive — safe on servers / inside venv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ── Project Imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet   import MobileNetV2UNet
from segmentation.loss         import DiceBCELoss
from segmentation.metrics      import compute_all_metrics
from segmentation.dataset_rsna import RSNAPneumoniaDataset


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — All knobs in one place
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "rsna_root":  str(PROJECT_ROOT / "data" / "rsna_pneumonia"),
    "train_csv":  str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "train_subset.csv"),
    "val_csv":    str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "val_subset.csv"),
    "results_dir":    str(PROJECT_ROOT / "results" / "model3c_final"),
    "checkpoint_best": str(PROJECT_ROOT / "results" / "model3c_final" / "model3c_best.pth"),
    "checkpoint_last": str(PROJECT_ROOT / "results" / "model3c_final" / "model3c_last.pth"),

    # ── Training Schedule ─────────────────────────────────────────────────────
    "max_epochs":      100,    # Hard upper limit — Early Stopping usually fires first
    "phase_b_start":   15,     # Switch from frozen → full fine-tune at this epoch

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "lr_phase_a":    1e-4,     # Learning rate while encoder is frozen
    "lr_phase_b":    1e-5,     # Lower LR for full fine-tuning (prevents forgetting)
    "weight_decay":  1e-5,     # L2 regularisation on all trainable parameters

    # ── Gradient Clipping ─────────────────────────────────────────────────────
    "grad_clip":     1.0,      # Max norm for gradient clipping (prevents explosions)

    # ── ReduceLROnPlateau Scheduler ───────────────────────────────────────────
    "sched_factor":   0.5,     # Multiply LR by this when val_dice plateaus
    "sched_patience": 7,       # Epochs of no improvement before LR is halved

    # ── Early Stopping ────────────────────────────────────────────────────────
    "early_stop_patience": 15, # Epochs of no improvement before training stops

    # ── DataLoader ────────────────────────────────────────────────────────────
    "batch_size":   16,
    "num_workers":  0,         # 0 = single-process (safe on all OS)

    # ── Misc ──────────────────────────────────────────────────────────────────
    "threshold":    0.5,       # Probability threshold for binary mask
    "random_seed":  42,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFEGUARD 1 — HEAVY AUGMENTATION
#  Applied only to training images. Validation always gets the clean transform.
#  Mask augmentations (flip, rotate) are handled INSIDE RSNAPneumoniaDataset
#  via synchronized_augment() — we only need torchvision transforms for the image.
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform():
    """
    Heavier augmentation pipeline for training:
      - ColorJitter:      Randomly vary brightness, contrast, and saturation.
                          Simulates different X-ray machine settings.
      - GaussianBlur:     Slight blur — simulates motion artefacts.
      - RandomErasing:    Randomly zeroes out a patch of the image.
                          Forces the model to not rely on any single region.
      - Normalization:    ImageNet stats required by MobileNetV2 encoder.
    Note: Spatial augmentations (flip, rotation) are applied inside the Dataset
    class synchronously on both image AND mask, so we do NOT add them here.
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # RandomErasing works on Tensors — applied AFTER ToTensor
        T.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
    ])

def get_val_transform():
    """Clean validation transforms — no augmentation."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  DEVICE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    elif preference == "mps" or (
        preference == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP — ONE EPOCH
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, max_epochs):
    """
    Run one pass over the training data.

    SAFEGUARD applied here:
      - Gradient Clipping: torch.nn.utils.clip_grad_norm_ keeps ||grad|| <= 1.0
        This prevents any single bad batch from causing a catastrophic weight update.

    Returns dict: train_loss, train_dice, train_iou, train_pixel_acc
    """
    model.train()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()

        # ── SAFEGUARD 5: Gradient Clipping ───────────────────────────────────
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])

        optimizer.step()

        with torch.no_grad():
            m = compute_all_metrics(preds, masks, threshold=CONFIG["threshold"])

        total_loss += loss.item()
        all_dice.append(m["dice"])
        all_iou.append(m["iou"])
        all_pix.append(m["pixel_acc"])

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
            print(f"    Step [{batch_idx+1:>3}/{len(loader)}]  "
                  f"Loss: {loss.item():.4f}  Dice: {m['dice']:.4f}")

    return {
        "train_loss":      total_loss / len(loader),
        "train_dice":      float(np.mean(all_dice)),
        "train_iou":       float(np.mean(all_iou)),
        "train_pixel_acc": float(np.mean(all_pix)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION LOOP — ONE EPOCH
# ═══════════════════════════════════════════════════════════════════════════════

def validate(model, loader, criterion, device):
    """Evaluate on validation set. No augmentation, no gradient updates."""
    model.eval()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)
            m     = compute_all_metrics(preds, masks, threshold=CONFIG["threshold"])

            total_loss += loss.item()
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_pix.append(m["pixel_acc"])

    return {
        "val_loss":      total_loss / len(loader),
        "val_dice":      float(np.mean(all_dice)),
        "val_iou":       float(np.mean(all_iou)),
        "val_pixel_acc": float(np.mean(all_pix)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(log_path: Path, results_dir: Path):
    """
    Reads training_logs.csv and saves 3 publication-ready plots:
      1. Loss curves (Train vs Val) — Overfitting is visible when val_loss rises
      2. Dice + IoU curves         — Primary metrics
      3. Train vs Val Dice gap     — Quantifies overfitting directly
    """
    import pandas as pd

    df     = pd.read_csv(log_path)
    epochs = df["epoch"].values

    plt.rcParams.update({
        "figure.facecolor": "#0F172A",
        "axes.facecolor":   "#1E293B",
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  "#CBD5E1",
        "axes.titlecolor":  "#F1F5F9",
        "axes.grid":        True,
        "grid.color":       "#334155",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "xtick.color":      "#94A3B8",
        "ytick.color":      "#94A3B8",
        "text.color":       "#F1F5F9",
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#334155",
        "legend.labelcolor":"#CBD5E1",
    })

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Model 3C — Full Fine-Tune Training Results\nFedMedSeg · RSNA Pneumonia",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Plot 1: Loss ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color="#38BDF8", lw=2, label="Train Loss")
    ax.plot(epochs, df["val_loss"],   color="#F87171", lw=2, ls="--", label="Val Loss")
    ax.set_title("Dice-BCE Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend()

    # Mark phase transition
    if "phase" in df.columns:
        phase_b_epoch = df.loc[df["phase"] == "B", "epoch"].min()
        if not np.isnan(phase_b_epoch):
            ax.axvline(phase_b_epoch, color="#FBBF24", lw=1.2, ls=":", alpha=0.8,
                       label=f"Phase B start (ep {int(phase_b_epoch)})")
            ax.legend()

    # ── Plot 2: Dice & IoU ────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, df["train_dice"], color="#4ADE80", lw=2, label="Train Dice")
    ax.plot(epochs, df["val_dice"],   color="#FB923C", lw=2, ls="--", label="Val Dice")
    ax.plot(epochs, df["val_iou"],    color="#A78BFA", lw=1.5, ls=":", label="Val IoU")
    ax.axhline(0.65, color="#FBBF24", lw=1, ls=":", alpha=0.8, label="Target (0.65)")
    ax.set_ylim(0, 1); ax.set_title("Dice & IoU")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.legend()

    # ── Plot 3: Overfitting Gap ───────────────────────────────────────────────
    ax = axes[2]
    gap = df["train_dice"].values - df["val_dice"].values
    ax.fill_between(epochs, 0, gap, alpha=0.4,
                    color=np.where(gap > 0.05, "#F87171", "#4ADE80").tolist()[0])
    ax.plot(epochs, gap, color="#F472B6", lw=2, label="Train - Val Dice Gap")
    ax.axhline(0.05, color="#FBBF24", lw=1.2, ls=":", label="Danger zone (>0.05)")
    ax.axhline(0.0,  color="#94A3B8", lw=0.8)
    ax.set_title("Overfitting Gap (lower = better)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (Train Dice − Val Dice)")
    ax.legend()

    plt.tight_layout()
    out = results_dir / "model3c_training_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Plots saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Model 3C Final Training")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint if found")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device      = get_device(args.device)
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = Path(CONFIG["checkpoint_best"])
    last_ckpt = Path(CONFIG["checkpoint_last"])

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n[Data] Loading RSNA dataset...")
    train_ds = RSNAPneumoniaDataset(
        rsna_root     = CONFIG["rsna_root"],
        subset_csv    = CONFIG["train_csv"],
        img_transform = get_train_transform(),
        augment       = True,    # Spatial augmentations (flip/rotate) inside dataset
    )
    val_ds = RSNAPneumoniaDataset(
        rsna_root     = CONFIG["rsna_root"],
        subset_csv    = CONFIG["val_csv"],
        img_transform = get_val_transform(),
        augment       = False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=device.type in ("cuda", "mps"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=device.type in ("cuda", "mps"),
    )
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model — Phase A: Encoder Frozen ───────────────────────────────────────
    print("\n[Model] Building MobileNetV2-UNet (Phase A: encoder frozen)...")
    model = MobileNetV2UNet(pretrained=True, freeze_encoder=True).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params — Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = DiceBCELoss(smooth=1e-6)

    # ── Phase A Optimizer & Schedulers ────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr_phase_a"], weight_decay=CONFIG["weight_decay"],
    )

    # ── SAFEGUARD 3: ReduceLROnPlateau ────────────────────────────────────────
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=CONFIG["sched_factor"],
        patience=CONFIG["sched_patience"],
    )

    # ── SAFEGUARD 4: CosineAnnealingLR (restarts each phase) ──────────────────
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["phase_b_start"],   # Cosine cycle = Phase A length
        eta_min=1e-6,
    )

    # ── CSV Logger ────────────────────────────────────────────────────────────
    log_path    = results_dir / "training_logs.csv"
    csv_headers = [
        "epoch", "phase",
        "train_loss", "val_loss",
        "train_dice", "val_dice",
        "train_iou",  "val_iou",
        "train_pixel_acc", "val_pixel_acc",
        "learning_rate", "epoch_time_sec",
        "early_stop_counter",
    ]

    start_epoch = 1
    best_val_dice      = 0.0
    early_stop_counter = 0
    phase              = "A"

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and last_ckpt.exists():
        print(f"\n[Resume] Loading checkpoint from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch        = ckpt.get("epoch", 1) + 1
        best_val_dice      = ckpt.get("best_val_dice", 0.0)
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        phase              = ckpt.get("phase", "A")
        print(f"  Resumed from epoch {start_epoch - 1} | Best val_dice: {best_val_dice:.4f}")
        # Append to existing log instead of overwriting
        log_file_mode = "a"
    else:
        log_file_mode = "w"

    if log_file_mode == "w":
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_headers)

    # Save config for reproducibility
    CONFIG["started_at"] = datetime.now().isoformat()
    CONFIG["device"]     = str(device)
    with open(results_dir / "training_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Model 3C — Full Fine-Tune Training")
    print(f"  Max epochs: {CONFIG['max_epochs']} | Early stop patience: {CONFIG['early_stop_patience']}")
    print(f"  Phase A (frozen):    epochs 1–{CONFIG['phase_b_start']-1}, LR={CONFIG['lr_phase_a']}")
    print(f"  Phase B (fine-tune): epochs {CONFIG['phase_b_start']}+,   LR={CONFIG['lr_phase_b']}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
        epoch_start = time.time()

        # ── SAFEGUARD 2: Phase Transition ─────────────────────────────────────
        if epoch == CONFIG["phase_b_start"] and phase == "A":
            phase = "B"
            print(f"\n{'='*55}")
            print(f"  PHASE B — Unfreezing FULL encoder (epoch {epoch})")
            print(f"  LR drops from {CONFIG['lr_phase_a']:.0e} → {CONFIG['lr_phase_b']:.0e}")
            print(f"{'='*55}\n")
            model.unfreeze_encoder(num_blocks=len(list(model.encoder_blocks.children())))

            # New optimizer with lower LR for fine-tuning
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CONFIG["lr_phase_b"], weight_decay=CONFIG["weight_decay"],
            )
            # Reset schedulers for Phase B
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max",
                factor=CONFIG["sched_factor"],
                patience=CONFIG["sched_patience"],
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["max_epochs"] - CONFIG["phase_b_start"],
                eta_min=1e-7,
            )

        # ── Train & Validate ──────────────────────────────────────────────────
        print(f"\n[Epoch {epoch:>3}/{CONFIG['max_epochs']}]  Phase {phase}")
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, CONFIG["max_epochs"])
        val_m   = validate(model, val_loader, criterion, device)

        # ── SAFEGUARD 3 & 4: Update both schedulers ────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        plateau_scheduler.step(val_m["val_dice"])
        cosine_scheduler.step()

        epoch_time = time.time() - epoch_start

        # ── Pretty epoch summary ───────────────────────────────────────────────
        gap = train_m["train_dice"] - val_m["val_dice"]
        gap_warn = " ⚠ GAP>0.1 (watch for overfitting)" if gap > 0.10 else ""
        print(f"\n  ┌──────────────────────────────────────────────────────────┐")
        print(f"  │  Epoch {epoch:>3}/{CONFIG['max_epochs']}  │  Phase {phase}  │  LR: {current_lr:.2e}            │")
        print(f"  ├───────────────────────────┬────────────┬────────────────┤")
        print(f"  │  Metric                   │   Train    │   Validation   │")
        print(f"  ├───────────────────────────┼────────────┼────────────────┤")
        print(f"  │  Loss (Dice-BCE)          │  {train_m['train_loss']:.4f}    │  {val_m['val_loss']:.4f}         │")
        print(f"  │  Dice Coefficient (↑)     │  {train_m['train_dice']:.4f}    │  {val_m['val_dice']:.4f}         │")
        print(f"  │  Mean IoU (↑)             │  {train_m['train_iou']:.4f}    │  {val_m['val_iou']:.4f}         │")
        print(f"  │  Pixel Accuracy (↑)       │  {train_m['train_pixel_acc']:.4f}    │  {val_m['val_pixel_acc']:.4f}         │")
        print(f"  │  Overfitting Gap (Train-Val Dice): {gap:.4f}{gap_warn}")
        print(f"  └──────────────────────────────────────────────────────────┘")
        print(f"  Time: {epoch_time:.1f}s")

        # ── Log to CSV ────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, phase,
                f"{train_m['train_loss']:.6f}",     f"{val_m['val_loss']:.6f}",
                f"{train_m['train_dice']:.6f}",     f"{val_m['val_dice']:.6f}",
                f"{train_m['train_iou']:.6f}",      f"{val_m['val_iou']:.6f}",
                f"{train_m['train_pixel_acc']:.6f}", f"{val_m['val_pixel_acc']:.6f}",
                f"{current_lr:.2e}", f"{epoch_time:.2f}",
                early_stop_counter,
            ])

        # ── SAFEGUARD 8: Checkpoint on new best ───────────────────────────────
        if val_m["val_dice"] > best_val_dice:
            best_val_dice      = val_m["val_dice"]
            early_stop_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ NEW BEST — val_dice = {best_val_dice:.4f}  → Saved to {best_ckpt.name}")
        else:
            early_stop_counter += 1
            remaining = CONFIG["early_stop_patience"] - early_stop_counter
            print(f"  No improvement. Early-stop counter: {early_stop_counter}/{CONFIG['early_stop_patience']}  ({remaining} left)")

        # Save last checkpoint (for --resume)
        torch.save({
            "epoch":              epoch,
            "model_state":        model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "best_val_dice":      best_val_dice,
            "early_stop_counter": early_stop_counter,
            "phase":              phase,
        }, last_ckpt)

        # ── SAFEGUARD 7: Early Stopping ───────────────────────────────────────
        if early_stop_counter >= CONFIG["early_stop_patience"]:
            print(f"\n  {'='*55}")
            print(f"  EARLY STOPPING — No improvement for {CONFIG['early_stop_patience']} epochs.")
            print(f"  Best val_dice achieved: {best_val_dice:.4f}  (epoch {epoch - early_stop_counter})")
            print(f"  {'='*55}\n")
            break

    # ═══════════════════════════════════════════════════════════════════════════
    #  POST-TRAINING
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Training Complete!")
    print(f"  Best Validation Dice : {best_val_dice:.4f}")
    print(f"  Best weights saved   : {best_ckpt}")
    print(f"{'='*65}\n")

    # Generate training plots
    print("[Plots] Generating training result charts...")
    make_plots(log_path, Path(CONFIG["results_dir"]))

    # Save evaluation summary JSON
    report = {
        "model":            "Model 3C — MobileNetV2-UNet Full Fine-Tune",
        "best_val_dice":    best_val_dice,
        "total_epochs_run": epoch,
        "early_stopped":    early_stop_counter >= CONFIG["early_stop_patience"],
        "safeguards": {
            "heavy_augmentation":   True,
            "two_phase_training":   True,
            "gradient_clipping":    CONFIG["grad_clip"],
            "l2_weight_decay":      CONFIG["weight_decay"],
            "reduce_lr_on_plateau": CONFIG["sched_patience"],
            "cosine_annealing":     True,
            "early_stopping":       CONFIG["early_stop_patience"],
            "checkpointing":        "best val_dice only",
        },
        "completed_at":     datetime.now().isoformat(),
        "config":           CONFIG,
    }
    report_path = Path(CONFIG["results_dir"]) / "model3c_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAll outputs saved to: {CONFIG['results_dir']}/")
    print(f"  ├── training_logs.csv")
    print(f"  ├── model3c_training_results.png   (3-panel chart)")
    print(f"  ├── model3c_best.pth               (best weights)")
    print(f"  ├── model3c_last.pth               (resume checkpoint)")
    print(f"  ├── training_config.json")
    print(f"  └── model3c_evaluation_report.json")


if __name__ == "__main__":
    main()
