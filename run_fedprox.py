"""
run_fedprox.py
==============
FedMedSeg Phase 3 — FedProx Experiment (Advanced Federated Learning)

PURPOSE:
  Demonstrate the FedProx algorithm, which improves upon FedAvg for
  Non-IID (heterogeneous) data distributions.

ALGORITHM (FedProx — Li et al., 2020):
  Same as FedAvg EXCEPT each client optimizes a MODIFIED loss function:

    L_prox(w_k) = L_task(w_k) + (μ/2) × ||w_k − w_global||²
                    ↑                          ↑
                 Standard loss            Proximal term
                 (Dice-BCE)          (prevents client drift)

  The proximal term penalizes the local model for deviating too far
  from the global model. This is like a "meeting moderator" that says:
  "You can learn from your local data, but don't stray too far from
  the group consensus."

WHY FedProx?
  In Non-IID settings, FedAvg can suffer from "client drift" where each
  client's local updates pull the global model in different directions.
  FedProx's regularization term stabilizes convergence.

  μ (mu) CONTROLS THE TRADE-OFF:
    μ = 0.0  → FedAvg (no regularization, clients are free to diverge)
    μ = 0.01 → Mild regularization (recommended starting point)
    μ = 0.1  → Strong regularization (clients stay very close to global)
    μ = 1.0  → Very strong (local training barely changes the model)

RUN:
    cd /home/rajat/Documents/Project/FedMedSeg
    .venv/bin/python run_fedprox.py

OPTIONAL FLAGS:
    --device cpu|cuda|mps    (default: auto-detect)
    --rounds 20              (default: 20)
    --local-epochs 1         (default: 1)
    --mu 0.01                (default: 0.01)
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
from segmentation.fl_server import create_fedprox_strategy


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
    "results_dir":    str(PROJECT_ROOT / "results" / "federated" / "fedprox"),
    "warm_start_ckpt": str(PROJECT_ROOT / "results" / "model3c_final" / "model3c_best.pth"),

    "num_rounds":     20,
    "local_epochs":   1,
    "mu":             0.01,    # FedProx proximal coefficient
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
#  CLIENT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_client_fn(
    client_configs: dict,
    device: torch.device,
    local_epochs: int,
    lr: float,
    mu: float,
):
    """
    Factory function for FedProx clients.

    The KEY DIFFERENCE from FedAvg: mu > 0, which activates the proximal
    term in the client's local training loss.
    """
    def client_fn(cid: str) -> fl.client.NumPyClient:
        cfg = client_configs[cid]

        # Build DataLoaders inside the worker — prevents Ray from pickling
        # large dataset objects into every actor (OOM fix).
        train_ds = RSNAPneumoniaDataset(
            rsna_root=cfg["rsna_root"],
            subset_csv=cfg["train_csv"],
            img_transform=get_train_transform(),
            augment=True,
        )
        val_ds = RSNAPneumoniaDataset(
            rsna_root=cfg["rsna_root"],
            subset_csv=cfg["val_csv"],
            img_transform=get_val_transform(),
            augment=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
        )

        model = MobileNetV2UNet(pretrained=False, freeze_encoder=False).to(device)

        return FedMedSegClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            client_name=cfg["name"],
            local_epochs=local_epochs,
            lr=lr,
            mu=mu,
        )

    return client_fn


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FedProx Experiment")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal coefficient (default: 0.01)")
    parser.add_argument("--warm-start", action="store_true",
                        help="Initialize from model3c_best.pth")
    args = parser.parse_args()

    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device = get_device(args.device)
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    num_rounds   = args.rounds
    local_epochs = args.local_epochs
    mu           = args.mu

    print("\n" + "=" * 65)
    print("  FedProx EXPERIMENT — Advanced Federated Learning")
    print(f"  Proximal coefficient μ = {mu}")
    print(f"  This prevents client drift on Non-IID data.")
    print(f"  Rounds: {num_rounds}  |  Local Epochs: {local_epochs}")
    print("=" * 65)

    # ── Data config: lightweight CSV paths only (DataLoaders built inside client_fn) ─
    print("\n[Data] Verifying dataset CSVs exist...")
    for csv_path in [CONFIG["client_a_csv"], CONFIG["client_b_csv"], CONFIG["val_csv"]]:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"  \u2713 {Path(csv_path).name}")

    client_configs = {
        "0": {
            "rsna_root":  CONFIG["rsna_root"],
            "train_csv":  CONFIG["client_a_csv"],
            "val_csv":    CONFIG["val_csv"],
            "batch_size": CONFIG["batch_size"],
            "name":       "Client A (Specialist)",
        },
        "1": {
            "rsna_root":  CONFIG["rsna_root"],
            "train_csv":  CONFIG["client_b_csv"],
            "val_csv":    CONFIG["val_csv"],
            "batch_size": CONFIG["batch_size"],
            "name":       "Client B (Clinic)",
        },
    }

    # ── Initial Global Model ──────────────────────────────────────────────────
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
    del init_model

    # ── Flower Strategy (FedProx) ─────────────────────────────────────────────
    strategy = create_fedprox_strategy(
        proximal_mu=mu,
        num_clients=2,
        initial_parameters=initial_parameters,
    )

    # ── CSV Logger ────────────────────────────────────────────────────────────
    log_path = results_dir / "round_metrics.csv"
    csv_headers = [
        "round", "global_val_dice", "global_val_iou", "global_val_pixel_acc",
        "train_dice", "train_iou", "round_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_headers)

    # ── Run Flower Simulation ─────────────────────────────────────────────────
    print(f"\n[FL] Starting FedProx simulation (μ={mu}, {num_rounds} rounds, "
          f"{len(client_configs)} clients)...\n")

    start_time = time.time()

    history = fl.simulation.start_simulation(
        client_fn=create_client_fn(client_configs, device, local_epochs, CONFIG["lr"], mu),
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        # num_cpus=4 → max 2 parallel actors (8 cores / 4 = 2), prevents OOM
        client_resources={"num_cpus": 4, "num_gpus": 0.0},
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

    round_metrics = []
    for rnd in range(1, num_rounds + 1):
        metrics_entry = {"round": rnd}

        if history.metrics_distributed:
            for key in ["val_dice", "val_iou", "val_pixel_acc"]:
                for r, val in history.metrics_distributed.get(key, []):
                    if r == rnd:
                        metrics_entry[f"global_{key}"] = val

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

    # Final metrics
    if round_metrics and "global_val_dice" in round_metrics[-1]:
        final_dice = round_metrics[-1].get("global_val_dice", 0)
        final_iou  = round_metrics[-1].get("global_val_iou", 0)
        final_pix  = round_metrics[-1].get("global_val_pixel_acc", 0)
    else:
        final_dice, final_iou, final_pix = 0, 0, 0

    # ── Save Report ───────────────────────────────────────────────────────────
    report = {
        "experiment":    "FedProx — Proximal Federated Learning",
        "algorithm":     "FedProx — Li et al., 2020",
        "purpose":       "Advanced FL with proximal term for Non-IID robustness",
        "proximal_mu":   mu,
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

    report_path = results_dir / "fedprox_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FedProx EXPERIMENT COMPLETE (μ = {mu})")
    print(f"{'='*65}")
    print(f"  Rounds:            {num_rounds}")
    print(f"  Proximal μ:        {mu}")
    print(f"  Total Time:        {total_time:.1f} sec")
    print(f"  Final Val Dice:    {final_dice:.4f}")
    print(f"  Final Val IoU:     {final_iou:.4f}")
    print(f"  Final Val PixAcc:  {final_pix:.4f}")
    print(f"\n  Results saved to: {results_dir}/")
    print(f"    ├── round_metrics.csv")
    print(f"    └── fedprox_report.json")


if __name__ == "__main__":
    main()
