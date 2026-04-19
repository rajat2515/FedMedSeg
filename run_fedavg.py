"""
run_fedavg.py
=============
FedMedSeg Phase 3 — Federated Averaging (FedAvg) Experiment

PURPOSE:
  Demonstrate standard Federated Learning using the FedAvg algorithm.
  Two hospital clients collaborate by sharing model weights (NOT data)
  through a central aggregation server.

ALGORITHM (FedAvg — McMahan et al., 2017):
  For each round r = 1, 2, ..., R:
    1. Server broadcasts global model w_r to all clients
    2. Each client k trains locally for E epochs on its Non-IID data
    3. Each client sends updated weights w_k back to the server
    4. Server aggregates:  w_{r+1} = Σ (n_k / n_total) × w_k
       where n_k = number of training samples on client k

  The key insight: DATA NEVER LEAVES THE HOSPITAL.
  Only model weights are transmitted — preserving patient privacy.

RUN:
    cd /home/rajat/Documents/Project/FedMedSeg
    .venv/bin/python run_fedavg.py

OPTIONAL FLAGS:
    --device cpu|cuda|mps    (default: auto-detect)
    --rounds 20              (default: 20)
    --local-epochs 1         (default: 1)
    --warm-start             (initialize from model3c_best.pth)
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
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import flwr as fl

# ── Project Imports ───────────────────────────────────────────────────────────
import os
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT / "src") + ":" + os.environ.get("PYTHONPATH", "")

from segmentation.model_unet import MobileNetV2UNet
from segmentation.loss import DiceBCELoss
from segmentation.metrics import compute_all_metrics
from segmentation.dataset_rsna import RSNAPneumoniaDataset
from segmentation.fl_client import (
    FedMedSegClient, get_parameters, set_parameters,
    evaluate_local,
)
from segmentation.fl_server import create_fedavg_strategy


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
    "results_dir":    str(PROJECT_ROOT / "results" / "federated" / "fedavg"),
    "warm_start_ckpt": str(PROJECT_ROOT / "results" / "model3c_final" / "model3c_best.pth"),

    "num_rounds":     20,
    "local_epochs":   1,
    "lr":             1e-4,
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
#  CLIENT FACTORY — Creates Flower Clients for Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def create_client_fn(
    client_configs: dict,
    device: torch.device,
    local_epochs: int,
    lr: float,
):
    """
    Factory function that Flower's simulation engine calls to create clients.

    Args:
        client_configs: Mapping from client ID (str) → {train_loader, val_loader, name}
        device: Compute device.
        local_epochs: Epochs per round.
        lr: Learning rate.

    Returns:
        Callable that creates a FedMedSegClient for a given client ID.
    """
    def client_fn(cid: str) -> fl.client.NumPyClient:
        cfg = client_configs[cid]
        # Each client gets a fresh model — weights will be set by the server
        model = MobileNetV2UNet(pretrained=True, freeze_encoder=False).to(device)

        return FedMedSegClient(
            model=model,
            train_loader=cfg["train_loader"],
            val_loader=cfg["val_loader"],
            device=device,
            client_name=cfg["name"],
            local_epochs=local_epochs,
            lr=lr,
            mu=0.0,  # FedAvg — no proximal term
        )

    return client_fn


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FedAvg Experiment")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--warm-start", action="store_true",
                        help="Initialize from model3c_best.pth")
    args = parser.parse_args()

    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device = get_device(args.device)
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    num_rounds = args.rounds
    local_epochs = args.local_epochs

    print("\n" + "=" * 65)
    print("  FEDERATED AVERAGING (FedAvg) EXPERIMENT")
    print("  2 hospital clients collaborate by sharing weights, NOT data.")
    print(f"  Rounds: {num_rounds}  |  Local Epochs: {local_epochs}")
    print("=" * 65)

    # ── Data Loaders ──────────────────────────────────────────────────────────
    print("\n[Data] Loading client datasets...")

    train_ds_a = RSNAPneumoniaDataset(
        rsna_root=CONFIG["rsna_root"],
        subset_csv=CONFIG["client_a_csv"],
        img_transform=get_train_transform(),
        augment=True,
    )
    train_ds_b = RSNAPneumoniaDataset(
        rsna_root=CONFIG["rsna_root"],
        subset_csv=CONFIG["client_b_csv"],
        img_transform=get_train_transform(),
        augment=True,
    )
    val_ds = RSNAPneumoniaDataset(
        rsna_root=CONFIG["rsna_root"],
        subset_csv=CONFIG["val_csv"],
        img_transform=get_val_transform(),
        augment=False,
    )

    train_loader_a = DataLoader(
        train_ds_a, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    train_loader_b = DataLoader(
        train_ds_b, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # ── Client configs for the factory ────────────────────────────────────────
    client_configs = {
        "0": {
            "train_loader": train_loader_a,
            "val_loader": val_loader,
            "name": "Client A (Specialist)",
        },
        "1": {
            "train_loader": train_loader_b,
            "val_loader": val_loader,
            "name": "Client B (Clinic)",
        },
    }

    # ── Initial Global Model Parameters ───────────────────────────────────────
    print("\n[Model] Preparing initial global model...")
    init_model = MobileNetV2UNet(pretrained=True, freeze_encoder=False)

    if args.warm_start:
        ckpt_path = CONFIG["warm_start_ckpt"]
        if Path(ckpt_path).exists():
            print(f"  Warm start: loading {ckpt_path}")
            init_model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True)
            )
        else:
            print(f"  ⚠ Warm start requested but {ckpt_path} not found. Using ImageNet weights.")

    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))
    del init_model  # Free memory

    # ── Flower Strategy ───────────────────────────────────────────────────────
    strategy = create_fedavg_strategy(
        num_clients=2,
        initial_parameters=initial_parameters,
    )

    # ── CSV Logger for Round Metrics ──────────────────────────────────────────
    log_path = results_dir / "round_metrics.csv"
    csv_headers = [
        "round", "global_val_dice", "global_val_iou", "global_val_pixel_acc",
        "train_dice", "train_iou", "round_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_headers)

    # ── Run Flower Simulation ─────────────────────────────────────────────────
    print(f"\n[FL] Starting FedAvg simulation ({num_rounds} rounds, {len(client_configs)} clients)...\n")

    start_time = time.time()

    history = fl.simulation.start_simulation(
        client_fn=create_client_fn(client_configs, device, local_epochs, CONFIG["lr"]),
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={
            "runtime_env": {
                "env_vars": {
                    "PYTHONPATH": str(PROJECT_ROOT / "src"),
                }
            }
        },
    )

    total_time = time.time() - start_time

    # ── Extract & Save Round Metrics ──────────────────────────────────────────
    print(f"\n[Results] Extracting metrics from {num_rounds} rounds...")

    # Extract distributed evaluation metrics
    round_metrics = []
    for rnd in range(1, num_rounds + 1):
        metrics_entry = {"round": rnd}

        # Distributed evaluation metrics
        if history.metrics_distributed:
            for key in ["val_dice", "val_iou", "val_pixel_acc"]:
                for r, val in history.metrics_distributed.get(key, []):
                    if r == rnd:
                        metrics_entry[f"global_{key}"] = val

        # Distributed fit metrics
        if history.metrics_distributed_fit:
            for key in ["train_dice", "train_iou"]:
                for r, val in history.metrics_distributed_fit.get(key, []):
                    if r == rnd:
                        metrics_entry[key] = val

        round_metrics.append(metrics_entry)

    # Write to CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for entry in round_metrics:
            writer.writerow({
                "round": entry.get("round", ""),
                "global_val_dice": f"{entry.get('global_val_dice', 0):.6f}",
                "global_val_iou": f"{entry.get('global_val_iou', 0):.6f}",
                "global_val_pixel_acc": f"{entry.get('global_val_pixel_acc', 0):.6f}",
                "train_dice": f"{entry.get('train_dice', 0):.6f}",
                "train_iou": f"{entry.get('train_iou', 0):.6f}",
                "round_time_sec": "",
            })

    # ── Final Global Model Evaluation ─────────────────────────────────────────
    print("\n[Eval] Evaluating final global model on validation set...")
    final_model = MobileNetV2UNet(pretrained=False, freeze_encoder=False).to(device)

    # Get final parameters from history
    if round_metrics and "global_val_dice" in round_metrics[-1]:
        final_dice = round_metrics[-1].get("global_val_dice", 0)
        final_iou  = round_metrics[-1].get("global_val_iou", 0)
        final_pix  = round_metrics[-1].get("global_val_pixel_acc", 0)
    else:
        final_dice, final_iou, final_pix = 0, 0, 0

    # ── Save Report ───────────────────────────────────────────────────────────
    report = {
        "experiment":    "Federated Averaging (FedAvg)",
        "algorithm":     "FedAvg — McMahan et al., 2017",
        "purpose":       "Standard federated learning — clients share weights, not data",
        "num_rounds":    num_rounds,
        "num_clients":   2,
        "local_epochs":  local_epochs,
        "warm_start":    args.warm_start,
        "total_time_sec": round(total_time, 2),
        "completed_at":  datetime.now().isoformat(),
        "final_metrics": {
            "val_dice":      final_dice,
            "val_iou":       final_iou,
            "val_pixel_acc": final_pix,
        },
        "round_history": round_metrics,
        "config":        CONFIG,
    }

    report_path = results_dir / "fedavg_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FedAvg EXPERIMENT COMPLETE")
    print(f"{'='*65}")
    print(f"  Rounds:            {num_rounds}")
    print(f"  Clients:           2")
    print(f"  Total Time:        {total_time:.1f} sec")
    print(f"  Final Val Dice:    {final_dice:.4f}")
    print(f"  Final Val IoU:     {final_iou:.4f}")
    print(f"  Final Val PixAcc:  {final_pix:.4f}")
    print(f"\n  Results saved to: {results_dir}/")
    print(f"    ├── round_metrics.csv")
    print(f"    └── fedavg_report.json")


if __name__ == "__main__":
    main()
