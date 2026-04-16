# src/segmentation/metrics.py
# FedMedSeg Phase 2 — Evaluation Metrics
#
# Three metrics are computed at pixel level:
#
#  1. DICE COEFFICIENT  (primary)
#     Dice = (2*TP) / (2*TP + FP + FN)
#     Range: [0, 1].  Higher = Better.  Target: > 0.65
#
#  2. INTERSECTION OVER UNION — IoU / Jaccard  (primary)
#     IoU  = TP / (TP + FP + FN)
#     Always lower than Dice for the same prediction.
#     Relationship:  IoU = Dice / (2 - Dice)
#
#  3. PIXEL ACCURACY  (secondary / sanity check)
#     PixelAcc = (TP + TN) / (TP + TN + FP + FN)
#     Note: Can be misleadingly high if most pixels are healthy.
#
# All functions accept THRESHOLDED binary tensors (0 or 1).
#
# TP = True Positive  (pred=1, gt=1)  ← correctly identified infected pixel
# TN = True Negative  (pred=0, gt=0)  ← correctly identified healthy pixel
# FP = False Positive (pred=1, gt=0)  ← falsely marked healthy pixel as infected
# FN = False Negative (pred=0, gt=1)  ← missed infected pixel

import torch


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Computes the Dice Coefficient (F1 score at pixel level).

    Formula:
        Dice = (2 * |P ∩ G| + ε) / (|P| + |G| + ε)

    Args:
        preds    (Tensor): Model output probabilities or logits. Shape: (B, 1, H, W)
        targets  (Tensor): Ground-truth binary masks.           Shape: (B, 1, H, W)
        threshold (float): Value above which a pixel is Pneumonia. Default 0.5.
        smooth   (float): Smoothing to avoid division by zero.

    Returns:
        float: Mean Dice score across the batch.
    """
    # Apply sigmoid if preds are logits (values outside [0,1])
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)

    # Threshold to binary
    preds_bin = (preds > threshold).float()

    # Flatten
    preds_flat   = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1).float()

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union        = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Computes Mean Intersection over Union (Jaccard Index).

    Formula:
        IoU = |P ∩ G| / (|P ∪ G|) = TP / (TP + FP + FN)

    Args:
        preds    (Tensor): Model output probabilities or logits. Shape: (B, 1, H, W)
        targets  (Tensor): Ground-truth binary masks.           Shape: (B, 1, H, W)
        threshold (float): Binarization threshold. Default 0.5.
        smooth   (float): Smoothing constant.

    Returns:
        float: Mean IoU across the batch.
    """
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)

    preds_bin = (preds > threshold).float()

    preds_flat   = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1).float()

    intersection = (preds_flat * targets_flat).sum(dim=1)                   # |P ∩ G|
    union        = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection  # |P ∪ G|

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Computes pixel-level accuracy.

    Formula:
        PixelAcc = (TP + TN) / (TP + TN + FP + FN)
                 = correct_pixels / total_pixels

    Args:
        preds    (Tensor): Model output probabilities or logits. Shape: (B, 1, H, W)
        targets  (Tensor): Ground-truth binary masks.           Shape: (B, 1, H, W)
        threshold (float): Binarization threshold. Default 0.5.

    Returns:
        float: Pixel accuracy as a fraction (0 to 1).
    """
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)

    preds_bin = (preds > threshold).float()
    correct   = (preds_bin == targets.float()).float()
    return correct.mean().item()


def compute_all_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Convenience function to compute all three metrics in one call.

    Returns:
        dict: {"dice": float, "iou": float, "pixel_acc": float}
    """
    return {
        "dice":      dice_coefficient(preds, targets, threshold),
        "iou":       mean_iou(preds, targets, threshold),
        "pixel_acc": pixel_accuracy(preds, targets, threshold),
    }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Perfect prediction: preds == targets
    targets  = torch.zeros(4, 1, 224, 224)
    targets[:, :, 50:100, 50:100] = 1.0   # White square = pneumonia region
    preds_perfect   = targets.clone()       # Perfect prediction
    preds_all_zero  = torch.zeros_like(targets)  # Lazy model (all healthy)

    print("=== Perfect Prediction ===")
    m = compute_all_metrics(preds_perfect, targets)
    print(f"  Dice:       {m['dice']:.4f}  (expected ≈ 1.0)")
    print(f"  IoU:        {m['iou']:.4f}  (expected ≈ 1.0)")
    print(f"  Pixel Acc:  {m['pixel_acc']:.4f}  (expected ≈ 1.0)")

    print("\n=== Lazy Model (predicts all-healthy) ===")
    m = compute_all_metrics(preds_all_zero, targets)
    print(f"  Dice:       {m['dice']:.4f}  (expected ≈ 0.0 — Dice punishes this)")
    print(f"  IoU:        {m['iou']:.4f}  (expected ≈ 0.0 — IoU punishes this)")
    print(f"  Pixel Acc:  {m['pixel_acc']:.4f}  (artificially high — 224*224-square / 224*224 total)")
    print("\nSanity check complete ✓")
