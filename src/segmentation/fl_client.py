# src/segmentation/fl_client.py
# FedMedSeg Phase 3 & 4 — Flower Federated Learning Client
#
# This module implements a Flower NumPyClient that bridges the gap between
# PyTorch model parameters and the NumPy arrays used by Flower for
# server ↔ client communication.
#
# SUPPORTED STRATEGIES:
#   - FedAvg:    Standard local training (criterion only)
#   - FedProx:   Adds proximal penalty to prevent client drift
#     L_prox = L_task + (μ/2) * Σ||w_local - w_global||²
#   - DP-FedProx: FedProx + Differential Privacy (Opacus DP-SGD)
#     Gradient clipping + Gaussian noise added before weight update
#
# COMMUNICATION FLOW:
#   1. Server sends global weights → set_parameters()
#   2. Client trains locally       → fit()  [optionally with DP]
#   3. Client returns new weights  → get_parameters()
#   4. Server evaluates globally   → evaluate()

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import flwr as fl

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet
from segmentation.loss import DiceBCELoss
from segmentation.metrics import compute_all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  OPACUS COMPATIBILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def disable_inplace_activations(model: nn.Module) -> nn.Module:
    """
    Disable all inplace activation functions in the model.

    MobileNetV2 uses ReLU6 (implemented as Hardtanh) with inplace=True.
    Opacus's per-sample gradient hooks conflict with inplace operations,
    causing a RuntimeError during the backward pass.
    This function walks the entire model and sets inplace=False on all
    ReLU, ReLU6, and Hardtanh activations.
    """
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardtanh)):
            module.inplace = False
    return model


def prepare_model_for_dp(model: nn.Module) -> nn.Module:
    """
    Full Opacus compatibility preparation:
      1. Replace BatchNorm2d → GroupNorm  (required for per-sample gradients)
      2. Disable inplace activations       (required for autograd hooks)
      3. Freeze unused encoder_blocks[18]  (never used in forward pass)

    MobileNetV2 has 19 feature blocks (indices 0–18). Our UNet decoder
    only uses blocks 0–17. Block 18 (1×1 conv: 320→1280ch) is the
    classification head pre-expansion and is NEVER called in forward().
    If its parameters remain trainable, Opacus registers them but
    backward() never computes their grad_sample → ValueError.
    """
    from opacus.validators import ModuleValidator
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    model = disable_inplace_activations(model)

    # Freeze encoder_blocks[18] — unused in forward(), causes Opacus to fail
    # if its trainable params have no grad_sample after backward().
    if hasattr(model, 'encoder_blocks') and len(model.encoder_blocks) > 18:
        for param in model.encoder_blocks[18].parameters():
            param.requires_grad = False

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETER HELPERS — PyTorch ↔ NumPy Conversion
# ═══════════════════════════════════════════════════════════════════════════════

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of NumPy arrays.

    This is what gets SENT to the server after local training.
    Flower requires NumPy format for serialization.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        List[np.ndarray]: One array per layer's weights/biases.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load server weights into the local model.

    This is called BEFORE local training to sync with the global model.

    Args:
        model (nn.Module): Local PyTorch model to update.
        parameters (List[np.ndarray]): New weights from the server.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCAL TRAINING — One Round of Client-Side Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-4,
    mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
    grad_clip: float = 1.0,
    # ── Phase 4: Differential Privacy ────────────────────────────────────────
    use_dp: bool = False,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    server_round: str = "?",
    client_name: str = "Client",
) -> Dict[str, float]:
    """
    Perform local training for one federated round.

    If mu > 0 and global_params is provided, adds the FedProx proximal term:
       L_total = L_task + (μ/2) * Σ||w_local - w_global||²

    If use_dp=True, wraps the model with Opacus PrivacyEngine (DP-SGD):
       - Per-sample gradients are clipped to max_grad_norm
       - Gaussian noise is added to the aggregated gradient
       - This guarantees (target_epsilon, target_delta)-DP

    Args:
        model (nn.Module): Local model to train.
        train_loader (DataLoader): Client's local training data.
        criterion (nn.Module): Loss function (DiceBCELoss).
        device (torch.device): Compute device.
        epochs (int): Number of local epochs per round. Default 1.
        lr (float): Learning rate for local SGD/Adam.
        mu (float): FedProx proximal coefficient. 0.0 = FedAvg.
        global_params (List[Tensor]): Frozen global weights for proximal term.
        grad_clip (float): Max gradient norm for clipping (non-DP mode).
        use_dp (bool): Enable Differential Privacy via Opacus. Default False.
        target_epsilon (float): Privacy budget ε. Only used if use_dp=True.
        target_delta (float): Privacy failure probability δ. Default 1e-5.
        max_grad_norm (float): Per-sample gradient clip norm for DP. Default 1.0.

    Returns:
        dict: {train_loss, train_dice, train_iou, train_pixel_acc}
              If use_dp=True, also includes {epsilon_spent, delta}.
    """
    model.train()

    # ── Phase 4: Differential Privacy Setup ──────────────────────────────────
    # When use_dp=True, Opacus wraps the model/optimizer/loader to intercept
    # gradients and add calibrated Gaussian noise (DP-SGD algorithm).
    # IMPORTANT: Opacus requires that the optimizer is created with ALL
    # trainable parameters (no filtering). The optimizer must be created
    # BEFORE make_private() so Opacus can wrap it properly.
    privacy_engine = None
    if use_dp:
        from opacus import GradSampleModule
        from segmentation.privacy import make_private

        # If this model was wrapped by a previous round's PrivacyEngine,
        # remove its hooks and unwrap it first to avoid double-wrapping errors.
        # NOTE: simply accessing ._module is NOT enough — the hooks are still
        # attached to the underlying model's parameters. remove_hooks() is required.
        if isinstance(model, GradSampleModule):
            model.remove_hooks()
            model = model._module

        optimizer = optim.Adam(
            model.parameters(),  # ALL params — Opacus needs full param list
            lr=lr,
            weight_decay=1e-5,
        )
        model, optimizer, train_loader, privacy_engine = make_private(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )

    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []
    num_batches = 0

    for epoch in range(epochs):
        # Add tqdm progress bar so the user can see training speed
        pbar = tqdm(train_loader, desc=f"[{client_name}] Round {server_round} | Epoch {epoch+1}/{epochs}", leave=True)
        
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, masks)

            # ── FedProx Proximal Term ─────────────────────────────────────
            # Only active when mu > 0 (FedProx mode)
            # When use_dp=False: add proximal term to loss before backward()
            # When use_dp=True:  CANNOT add to loss — Opacus computes per-sample
            #   gradients via forward-pass hooks, but the proximal term bypasses
            #   those hooks (it's a direct param→param operation), leaving
            #   .grad_sample uninitialized → ValueError.
            #   Fix: run task loss backward first (Opacus hooks fire cleanly),
            #   then manually add the proximal gradient to .grad afterward.
            if mu > 0.0 and global_params is not None and not use_dp:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    model.parameters(), global_params
                ):
                    proximal_term += (
                        (local_param - global_param).norm(2) ** 2
                    )
                loss = loss + (mu / 2.0) * proximal_term

            loss.backward()

            # When DP is on, manually inject proximal gradient AFTER backward()
            # so Opacus hooks have already fired and .grad_sample is populated.
            if use_dp and mu > 0.0 and global_params is not None:
                with torch.no_grad():
                    for local_param, global_param in zip(
                        model.parameters(), global_params
                    ):
                        if local_param.requires_grad and local_param.grad is not None:
                            local_param.grad += mu * (local_param.data - global_param)

            if not use_dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                m = compute_all_metrics(preds, masks, threshold=0.5)

            total_loss += loss.item()
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_pix.append(m["pixel_acc"])
            num_batches += 1
            
            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{m['dice']:.4f}"})

    result = {
        "train_loss":      total_loss / max(num_batches, 1),
        "train_dice":      float(np.mean(all_dice)),
        "train_iou":       float(np.mean(all_iou)),
        "train_pixel_acc": float(np.mean(all_pix)),
    }

    # ── Report privacy budget consumed this round ─────────────────────────────
    if use_dp and privacy_engine is not None:
        from segmentation.privacy import get_privacy_spent
        epsilon_spent, delta = get_privacy_spent(privacy_engine)
        result["epsilon_spent"] = epsilon_spent
        result["delta"]         = delta

    # ── Cleanup: free GPU memory after DP training ────────────────────────────
    # Opacus per-sample gradient hooks hold large intermediate tensors.
    # Explicitly delete them and clear CUDA cache to prevent OOM when the
    # next client trains on the same GPU.
    # CRITICAL: call remove_hooks() on the GradSampleModule BEFORE deleting the
    # privacy engine. Without this, the hooks remain registered on the underlying
    # model's parameters, and the next call to make_private() raises:
    #   ValueError: Trying to add hooks twice to the same model
    if use_dp:
        if hasattr(model, 'remove_hooks'):   # model is GradSampleModule here
            model.remove_hooks()
        del optimizer, privacy_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCAL EVALUATION — Evaluate Global Model on Client's Local Data
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_local(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on a validation set.

    Called by Flower to check global model performance from each client's
    perspective.

    Args:
        model (nn.Module): Model with current global weights.
        val_loader (DataLoader): Validation DataLoader.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.

    Returns:
        Tuple[float, dict]: (loss, {dice, iou, pixel_acc})
    """
    model.eval()
    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)
            m     = compute_all_metrics(preds, masks, threshold=0.5)

            total_loss += loss.item()
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_pix.append(m["pixel_acc"])

    num_batches = max(len(all_dice), 1)
    metrics = {
        "val_dice":      float(np.mean(all_dice)),
        "val_iou":       float(np.mean(all_iou)),
        "val_pixel_acc": float(np.mean(all_pix)),
    }
    return total_loss / num_batches, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  FLOWER CLIENT — The "Hospital" in Federated Learning
# ═══════════════════════════════════════════════════════════════════════════════

class FedMedSegClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for the FedMedSeg segmentation task.

    Each instance represents one "hospital" participating in federation.
    The hospital has its own local Non-IID dataset but collaborates by
    sharing model weights (not data) with the central server.

    Args:
        model (nn.Module): Local MobileNetV2-UNet model.
        train_loader (DataLoader): Client's local training data.
        val_loader (DataLoader): Shared global validation data.
        device (torch.device): Compute device.
        client_name (str): Human-readable name for logging.
        local_epochs (int): Training epochs per federated round.
        lr (float): Local learning rate.
        mu (float): FedProx proximal coefficient. 0.0 = FedAvg.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        client_name: str = "Client",
        local_epochs: int = 1,
        lr: float = 1e-4,
        mu: float = 0.0,
        lock = None,
        # ── Phase 4: DP args ─────────────────────────────────────────────────
        use_dp: bool = False,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.client_name    = client_name
        self.local_epochs   = local_epochs
        self.lr             = lr
        self.mu             = mu
        self.lock           = lock
        self.use_dp         = use_dp
        self.target_epsilon = target_epsilon
        self.target_delta   = target_delta
        self.max_grad_norm  = max_grad_norm
        self.criterion      = DiceBCELoss(smooth=1e-6)

        if self.use_dp:
            self.model = prepare_model_for_dp(self.model)

    def get_parameters(self, config: Dict = None) -> List[np.ndarray]:
        """Send local model parameters to the server."""
        return get_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global weights → train locally → return updated weights.

        This is the core of federated learning:
          1. Update local model with server's global weights
          2. Train on local Non-IID data for `local_epochs`
          3. Return new weights + number of training samples + metrics
        """
        # Step 1: Sync with global model
        set_parameters(self.model, parameters)

        # Step 2: Store global params for FedProx proximal term
        global_params = None
        if self.mu > 0.0:
            global_params = [
                param.detach().clone()
                for param in self.model.parameters()
            ]

        # Step 3: Local training (Serialized if lock is provided)
        if self.lock:
            print(f"  [{self.client_name}] Waiting for GPU access...")
            self.lock.acquire()
            print(f"  [{self.client_name}] Acquired GPU lock, starting training.")
            
        try:
            metrics = train_local(
                model=self.model,
                train_loader=self.train_loader,
                criterion=self.criterion,
                device=self.device,
                epochs=self.local_epochs,
                lr=self.lr,
                mu=self.mu,
                global_params=global_params,
                # Phase 4: DP args
                use_dp=self.use_dp,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
                server_round=str(config.get("server_round", "?")),
                client_name=self.client_name,
            )
        finally:
            if self.lock:
                self.lock.release()
                print(f"  [{self.client_name}] Released GPU lock.")

        eps_str = (
            f"  ε={metrics['epsilon_spent']:.2f}"
            if self.use_dp and "epsilon_spent" in metrics
            else ""
        )
        print(f"  [{self.client_name}] fit() → "
              f"Loss: {metrics['train_loss']:.4f}  "
              f"Dice: {metrics['train_dice']:.4f}"
              f"{eps_str}")

        # Return: (updated weights, num samples, metrics dict)
        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            metrics,
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the global model on validation data.

        The server calls this to check how the aggregated model performs
        from this client's perspective.
        """
        set_parameters(self.model, parameters)
        
        if self.lock:
            self.lock.acquire()
        try:
            loss, metrics = evaluate_local(
                model=self.model,
                val_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )
        finally:
            if self.lock:
                self.lock.release()

        print(f"  [{self.client_name}] evaluate() → "
              f"Val Dice: {metrics['val_dice']:.4f}  "
              f"Val IoU: {metrics['val_iou']:.4f}")

        return (
            float(loss),
            len(self.val_loader.dataset),
            metrics,
        )
