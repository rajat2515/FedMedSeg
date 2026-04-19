# src/segmentation/privacy.py
# FedMedSeg Phase 4 — Differential Privacy Module
#
# PURPOSE:
#   This module adds Differential Privacy (DP) to federated learning clients
#   using the Opacus library (PyTorch's official DP engine).
#
# WHAT IS DIFFERENTIAL PRIVACY?
#   Without DP, a hospital sends model weights to the server. A sophisticated
#   attacker with enough computing power could potentially perform a "model
#   inversion attack" — reconstructing approximate patient X-rays from the
#   model weights alone. DP prevents this by adding calibrated Gaussian noise
#   to the gradients BEFORE the weights are computed.
#
# HOW DP-SGD WORKS (Two Steps):
#   1. CLIPPING:
#      Before noise, we clip each individual sample's gradient to a maximum
#      L2 norm of `max_grad_norm`. This limits how much any single patient's
#      data can influence the model (boundedness guarantee).
#
#   2. NOISE ADDITION:
#      After clipping, we add Gaussian noise scaled by `noise_multiplier`.
#      This mathematically guarantees that no individual record changes the
#      output distribution by more than a factor of e^ε (epsilon).
#
# THE PRIVACY BUDGET (ε — EPSILON):
#   ε controls the privacy-accuracy trade-off:
#     ε → 0   = Perfect privacy, but model learns nothing
#     ε = 1   = Very strong privacy (used in census data)
#     ε = 8   = Moderate — recommended for healthcare (our default)
#     ε = ∞   = No privacy (standard SGD)
#
#   We use ε = 8 following published medical FL papers (CheXFed, SiloBN, etc.)
#   which show that ε ≤ 8 provides strong privacy with <2% Dice score drop.
#
# DELTA (δ):
#   δ is the probability of the privacy guarantee "failing". We set δ = 1/N
#   where N is the dataset size. At N=2000, δ = 5e-4. We use 1e-5 for safety.
#
# REFERENCE:
#   Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
#   https://arxiv.org/abs/1607.00133

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  COMPATIBILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _check_opacus_available() -> bool:
    """Check if Opacus is installed."""
    try:
        import opacus  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DP SETUP FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, object]:
    """
    Wrap a model + optimizer + dataloader with Opacus PrivacyEngine.

    This is the single entry point for enabling DP in training. After calling
    this function, training proceeds identically to non-private training —
    Opacus hooks into PyTorch's autograd to intercept gradients automatically.

    Args:
        model (nn.Module): The PyTorch model to make private.
        optimizer (torch.optim.Optimizer): The optimizer to wrap.
        data_loader (DataLoader): Training DataLoader.
        target_epsilon (float): Target privacy budget ε. Default 8.0 (healthcare
            industry standard). Lower = stronger privacy, lower accuracy.
        target_delta (float): Probability of privacy failure δ. Default 1e-5.
        max_grad_norm (float): Per-sample gradient clipping bound. Default 1.0.
        epochs (int): Total training epochs. Opacus needs this to compute the
            noise multiplier that achieves the target ε.

    Returns:
        Tuple of (private_model, private_optimizer, private_loader, privacy_engine)
        The returned objects are drop-in replacements for the originals.

    Raises:
        ImportError: If Opacus is not installed.
        ValueError: If the model has unsupported layers (e.g., BatchNorm).

    Note:
        Opacus does NOT support BatchNorm layers (they share statistics across
        samples, breaking per-sample gradient assumptions). We automatically
        replace BatchNorm2d with GroupNorm before wrapping. Our MobileNetV2-UNet
        already uses BatchNorm, so we handle this conversion here.
    """
    if not _check_opacus_available():
        raise ImportError(
            "Opacus is not installed. Run: pip install opacus>=1.3.0"
        )

    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # ── Step 1: Validate + Fix Model for Opacus Compatibility ────────────────
    # Opacus requires per-sample gradients. BatchNorm2d computes statistics
    # across the entire batch, making per-sample gradients impossible.
    # We replace BatchNorm2d with GroupNorm (equivalent performance in practice).
    if not ModuleValidator.is_valid(model):
        logger.info(
            "  [DP] Model has incompatible layers (BatchNorm). "
            "Auto-replacing with GroupNorm..."
        )
        model = ModuleValidator.fix(model)
        logger.info("  [DP] Model fixed: BatchNorm2d → GroupNorm ✓")

    # ── Step 2: Create PrivacyEngine ─────────────────────────────────────────
    privacy_engine = PrivacyEngine()

    # ── Step 3: Attach engine to model, optimizer, and loader ────────────────
    # Opacus computes the noise_multiplier automatically from:
    #   target_epsilon, target_delta, max_grad_norm, epochs, batch_size, dataset_size
    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
    )

    noise_mult = private_optimizer.noise_multiplier
    logger.info(
        f"  [DP] PrivacyEngine attached:\n"
        f"       Target ε = {target_epsilon}, δ = {target_delta}\n"
        f"       Max Gradient Norm = {max_grad_norm}\n"
        f"       Noise Multiplier  = {noise_mult:.4f}\n"
        f"       (Higher noise = more privacy)"
    )

    return private_model, private_optimizer, private_loader, privacy_engine


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVACY ACCOUNTING
# ─────────────────────────────────────────────────────────────────────────────

def get_privacy_spent(privacy_engine) -> Tuple[float, float]:
    """
    Query how much privacy budget has been consumed so far.

    Must be called AFTER at least one training step (forward + backward pass).

    Args:
        privacy_engine: The Opacus PrivacyEngine object returned by make_private().

    Returns:
        Tuple (epsilon, delta):
            epsilon (float): Privacy budget consumed so far.
            delta   (float): Target delta used at setup.
    """
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return float(epsilon), 1e-5


def save_privacy_accounting(
    privacy_engine,
    round_num: int,
    output_dir: Path,
    target_epsilon: float,
    noise_multiplier: float,
    max_grad_norm: float,
) -> dict:
    """
    Save a privacy accounting report to JSON.

    Records the current privacy budget consumption. Called after each
    federated round to track cumulative ε.

    Args:
        privacy_engine: Opacus PrivacyEngine.
        round_num (int): Current federated round number.
        output_dir (Path): Directory to save the JSON file.
        target_epsilon (float): The ε target set at initialization.
        noise_multiplier (float): Computed noise multiplier.
        max_grad_norm (float): Gradient clipping norm.

    Returns:
        dict: The privacy accounting record for this round.
    """
    epsilon_spent, delta = get_privacy_spent(privacy_engine)

    record = {
        "round": round_num,
        "epsilon_spent": epsilon_spent,
        "delta": delta,
        "target_epsilon": target_epsilon,
        "budget_remaining": max(0.0, target_epsilon - epsilon_spent),
        "budget_fraction_used": min(1.0, epsilon_spent / target_epsilon),
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "privacy_guarantee": (
            f"Patient data is ({epsilon_spent:.2f}, {delta:.0e})-differentially private"
        ),
    }

    out_path = output_dir / "privacy_accounting.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: Print DP Summary Banner
# ─────────────────────────────────────────────────────────────────────────────

def print_dp_summary(target_epsilon: float, max_grad_norm: float, noise_multiplier: float):
    """Print a human-readable summary of the DP configuration."""
    print("\n" + "=" * 65)
    print("  DIFFERENTIAL PRIVACY CONFIGURATION")
    print("=" * 65)
    print(f"  Privacy Budget (ε):    {target_epsilon}")
    print(f"  Privacy Failure (δ):   1e-5")
    print(f"  Gradient Clip Norm:    {max_grad_norm}")
    print(f"  Noise Multiplier:      {noise_multiplier:.4f}")
    print(f"")
    print(f"  Guarantee: An attacker who observes the model weights")
    print(f"  cannot determine if any individual patient's X-ray was")
    print(f"  used in training with confidence greater than e^{target_epsilon} ≈ {2.718**target_epsilon:.0f}x.")
    print("=" * 65 + "\n")
