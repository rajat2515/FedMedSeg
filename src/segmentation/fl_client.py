# src/segmentation/fl_client.py
# FedMedSeg Phase 3 — Flower Federated Learning Client
#
# This module implements a Flower NumPyClient that bridges the gap between
# PyTorch model parameters and the NumPy arrays used by Flower for
# server ↔ client communication.
#
# SUPPORTED STRATEGIES:
#   - FedAvg:  Standard local training (criterion only)
#   - FedProx: Adds proximal penalty to prevent client drift
#     L_prox = L_task + (μ/2) * Σ||w_local - w_global||²
#
# COMMUNICATION FLOW:
#   1. Server sends global weights → set_parameters()
#   2. Client trains locally       → fit()
#   3. Client returns new weights  → get_parameters()
#   4. Server evaluates globally   → evaluate()

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
) -> Dict[str, float]:
    """
    Perform local training for one federated round.

    If mu > 0 and global_params is provided, adds the FedProx proximal term:
       L_total = L_task + (μ/2) * Σ||w_local - w_global||²

    This penalizes the local model for drifting too far from the global model,
    which is critical for Non-IID data stability.

    Args:
        model (nn.Module): Local model to train.
        train_loader (DataLoader): Client's local training data.
        criterion (nn.Module): Loss function (DiceBCELoss).
        device (torch.device): Compute device.
        epochs (int): Number of local epochs per round. Default 1.
        lr (float): Learning rate for local SGD/Adam.
        mu (float): FedProx proximal coefficient. 0.0 = FedAvg.
        global_params (List[Tensor]): Frozen global weights for proximal term.
        grad_clip (float): Max gradient norm for clipping.

    Returns:
        dict: {train_loss, train_dice, train_iou, train_pixel_acc}
    """
    model.train()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )

    total_loss = 0.0
    all_dice, all_iou, all_pix = [], [], []
    num_batches = 0

    for epoch in range(epochs):
        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, masks)

            # ── FedProx Proximal Term ─────────────────────────────────────
            # Only active when mu > 0 (FedProx mode)
            # Penalizes local weights for deviating from global weights
            if mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    model.parameters(), global_params
                ):
                    proximal_term += (
                        (local_param - global_param).norm(2) ** 2
                    )
                loss = loss + (mu / 2.0) * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                m = compute_all_metrics(preds, masks, threshold=0.5)

            total_loss += loss.item()
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_pix.append(m["pixel_acc"])
            num_batches += 1

    return {
        "train_loss":      total_loss / max(num_batches, 1),
        "train_dice":      float(np.mean(all_dice)),
        "train_iou":       float(np.mean(all_iou)),
        "train_pixel_acc": float(np.mean(all_pix)),
    }


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
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.client_name  = client_name
        self.local_epochs = local_epochs
        self.lr           = lr
        self.mu           = mu
        self.criterion    = DiceBCELoss(smooth=1e-6)

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

        # Step 3: Local training
        metrics = train_local(
            model=self.model,
            train_loader=self.train_loader,
            criterion=self.criterion,
            device=self.device,
            epochs=self.local_epochs,
            lr=self.lr,
            mu=self.mu,
            global_params=global_params,
        )

        print(f"  [{self.client_name}] fit() → "
              f"Loss: {metrics['train_loss']:.4f}  "
              f"Dice: {metrics['train_dice']:.4f}")

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
        loss, metrics = evaluate_local(
            model=self.model,
            val_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device,
        )

        print(f"  [{self.client_name}] evaluate() → "
              f"Val Dice: {metrics['val_dice']:.4f}  "
              f"Val IoU: {metrics['val_iou']:.4f}")

        return (
            float(loss),
            len(self.val_loader.dataset),
            metrics,
        )
