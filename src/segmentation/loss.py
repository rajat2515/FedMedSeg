# src/segmentation/loss.py
# FedMedSeg Phase 2 — Loss Functions
#
# WHY DICE + BCE?
#   - BCE: Penalizes each pixel individually.  Good for sharp gradients.
#   - Dice: Penalizes poor *region overlap*.  Prevents the "lazy model" that just
#           predicts all-healthy because healthy pixels dominate the image.
#   Together:  L_total = L_BCE  +  L_Dice   (Dice-BCE Hybrid)
#
# FORMULA (Soft Dice):
#   Dice(P, G) = (2 * sum(p_i * g_i) + ε) / (sum(p_i) + sum(g_i) + ε)
#   L_Dice     = 1 - Dice(P, G)
#
# ε (epsilon) = 1e-6  →  prevents division by zero when both pred & gt are zero.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    Loss = 1 - Dice

    Args:
        smooth (float): Laplacian smoothing constant. Default 1e-6.
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   (Tensor): Raw logits from the model. Shape: (B, 1, H, W)
            targets (Tensor): Binary ground-truth masks.  Shape: (B, 1, H, W)

        Returns:
            Tensor: Scalar Dice loss value.
        """
        # Convert logits to probabilities
        preds = torch.sigmoid(preds)

        # Flatten spatial dimensions for calculation
        preds_flat   = preds.view(preds.size(0), -1)    # (B, H*W)
        targets_flat = targets.view(targets.size(0), -1) # (B, H*W)

        # Soft Dice numerator and denominator per sample in the batch
        intersection = (preds_flat * targets_flat).sum(dim=1)           # (B,)
        union        = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)  # (B,)

        # Dice coefficient per sample, then average over batch
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Hybrid Loss = BCE Loss + Dice Loss.

    This is the gold standard for medical image segmentation.
    - BCE handles pixel-level accuracy with strong gradients.
    - Dice handles regional overlap and combats class imbalance.

    Total Loss:  L = BCE(logits, targets) + (1 - Dice(sigmoid(logits), targets))

    Args:
        smooth (float): Smoothing constant passed to DiceLoss. Default 1e-6.
        bce_weight (float): Weight applied to the BCE term. Default 1.0.
        dice_weight (float): Weight applied to the Dice term. Default 1.0.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super(DiceBCELoss, self).__init__()
        self.bce_loss  = nn.BCEWithLogitsLoss()   # expects raw logits
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   (Tensor): Raw logits from the model. Shape: (B, 1, H, W)
            targets (Tensor): Binary masks (float32).    Shape: (B, 1, H, W)

        Returns:
            Tensor: Scalar hybrid loss.
        """
        bce  = self.bce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, C, H, W = 4, 1, 224, 224
    preds   = torch.randn(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W)).float()

    criterion = DiceBCELoss()
    loss = criterion(preds, targets)

    print(f"[DiceBCELoss] Input logits shape : {preds.shape}")
    print(f"[DiceBCELoss] Target mask  shape : {targets.shape}")
    print(f"[DiceBCELoss] Loss value         : {loss.item():.4f}  ✓")
    assert loss.item() > 0, "Loss should be positive for random inputs."
    print("Sanity check passed!")
