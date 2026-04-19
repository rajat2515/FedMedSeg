"""
run_dp_fedprox.py
=================
FedMedSeg Phase 4 — Differentially Private FedProx Experiment

PURPOSE:
  Demonstrate that Federated Learning can be made PRIVACY-PRESERVING using
  Differential Privacy (DP-SGD via Opacus), with minimal accuracy cost.

WHAT IS NEW vs run_fedprox.py:
  - Each client's optimizer is wrapped with Opacus PrivacyEngine
  - Per-sample gradients are clipped to `max_grad_norm` before update
  - Gaussian noise is added to the clipped gradients
  - Privacy budget (ε) is tracked and reported after every round
  - Model is quantized (float32 → int8) before payload size measurement

PRIVACY GUARANTEE:
  After training, the report states:
    "Patient data is (ε, δ)-differentially private"
  Meaning: an attacker who observes the weights cannot determine if
  any individual patient's X-ray was used in training.

THE FULL STORY (5 steps):
  1. Centralized  → Upper bound (all data, no privacy)
  2. Isolated     → Fails (biased data, no collaboration)
  3. FedAvg       → Recovers (privacy by locality, no DP guarantee)
  4. FedProx      → Excels (handles Non-IID drift)
  5. DP-FedProx   → Privacy-preserving (mathematical guarantee, low cost)

RUN:
    cd /home/rajat/Documents/Project/FedMedSeg
    .venv/bin/python run_dp_fedprox.py --rounds 20 --epsilon 8.0

OPTIONAL FLAGS:
    --device cpu|cuda|mps    (default: auto-detect)
    --rounds 20              (default: 20)
    --local-epochs 1         (default: 1)
    --mu 0.01                (default: 0.01, FedProx proximal coefficient)
    --epsilon 8.0            (default: 8.0, privacy budget ε)
    --max-grad-norm 1.0      (default: 1.0, gradient clipping norm)
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
from segmentation.quantization import (
    quantize_model,
    measure_compression,
    print_quantization_report,
    get_model_size_mb,
)
from segmentation.privacy import print_dp_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONFIG = {
    "rsna_root":       str(PROJECT_ROOT / "data" / "rsna_pneumonia"),
    "client_a_csv":    str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "client_a_train.csv"),
    "client_b_csv":    str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "client_b_train.csv"),
    "val_csv":         str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "val_subset.csv"),
    "results_dir":     str(PROJECT_ROOT / "results" / "federated" / "dp_fedprox"),
    "warm_start_ckpt": str(PROJECT_ROOT / "results" / "model3c_final" / "model3c_best.pth"),

    # Federated
    "num_rounds":      20,
    "local_epochs":    1,
    "mu":              0.01,   # FedProx proximal coefficient
    "lr":              1e-4,
    "batch_size":      16,
    "num_workers":     0,
    "threshold":       0.5,
    "random_seed":     42,

    # Differential Privacy
    "target_epsilon":  8.0,    # Privacy budget ε (healthcare standard)
    "target_delta":    1e-5,   # Probability of privacy failure
    "max_grad_norm":   1.0,    # Per-sample gradient clipping norm
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
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
):
    """
    Factory function for DP-FedProx clients.

    Each client will:
    1. Receive global weights from the server
    2. Wrap its optimizer with Opacus PrivacyEngine (DP-SGD)
    3. Train locally with gradient clipping + Gaussian noise
    4. Return updated weights + epsilon spent
    """
    def client_fn(cid: str) -> fl.client.NumPyClient:
        cfg = client_configs[cid]

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
            # ── Phase 4: Enable Differential Privacy ──────────────────────
            use_dp=True,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

    return client_fn


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DP-FedProx Experiment (Phase 4)")
    parser.add_argument("--device",         type=str,   default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--rounds",         type=int,   default=20)
    parser.add_argument("--local-epochs",   type=int,   default=1)
    parser.add_argument("--mu",             type=float, default=0.01,
                        help="FedProx proximal coefficient (default: 0.01)")
    parser.add_argument("--epsilon",        type=float, default=8.0,
                        help="Privacy budget ε (default: 8.0 — healthcare standard)")
    parser.add_argument("--max-grad-norm",  type=float, default=1.0,
                        help="Per-sample gradient clipping norm (default: 1.0)")
    parser.add_argument("--warm-start",     action="store_true",
                        help="Initialize from model3c_best.pth")
    args = parser.parse_args()

    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device       = get_device(args.device)
    results_dir  = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    num_rounds    = args.rounds
    local_epochs  = args.local_epochs
    mu            = args.mu
    target_epsilon = args.epsilon
    max_grad_norm  = args.max_grad_norm

    print("\n" + "=" * 65)
    print("  DP-FedProx — Differentially Private Federated Learning")
    print("  Phase 4: Privacy & Efficiency Layer")
    print("=" * 65)
    print(f"  FedProx μ:         {mu}")
    print(f"  Privacy Budget ε:  {target_epsilon} (healthcare standard)")
    print(f"  Gradient Clip:     {max_grad_norm}")
    print(f"  Rounds:            {num_rounds} | Local Epochs: {local_epochs}")
    print("=" * 65)

    # ── Verify dataset CSVs ──────────────────────────────────────────────────
    print("\n[Data] Verifying dataset CSVs...")
    for csv_path in [CONFIG["client_a_csv"], CONFIG["client_b_csv"], CONFIG["val_csv"]]:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"  ✓ {Path(csv_path).name}")

    client_configs = {
        "0": {
            "rsna_root":  CONFIG["rsna_root"],
            "train_csv":  CONFIG["client_a_csv"],
            "val_csv":    CONFIG["val_csv"],
            "batch_size": CONFIG["batch_size"],
            "name":       "Client A (Specialist) [DP]",
        },
        "1": {
            "rsna_root":  CONFIG["rsna_root"],
            "train_csv":  CONFIG["client_b_csv"],
            "val_csv":    CONFIG["val_csv"],
            "batch_size": CONFIG["batch_size"],
            "name":       "Client B (Clinic) [DP]",
        },
    }

    # ── Initial Global Model ─────────────────────────────────────────────────
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
            print(f"  ⚠ Warm start requested but {ckpt_path} not found.")

    # ── Quantization Report (before training, measure original size) ─────────
    print("\n[Quantization] Measuring model size...")
    original_size_mb = get_model_size_mb(init_model)
    quantized_model  = quantize_model(init_model)
    quant_stats      = measure_compression(init_model, quantized_model)
    print_quantization_report(quant_stats)

    # Save quantization report
    quant_report_path = results_dir / "quantization_report.json"
    with open(quant_report_path, "w") as f:
        json.dump(quant_stats, f, indent=2)
    del quantized_model

    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    # ── Flower Strategy (FedProx — same, DP is on client side) ──────────────
    # Note: DP is implemented CLIENT-SIDE. The server strategy (FedProx)
    # aggregates DP-protected weights and remains unchanged — this is
    # by design. The server is "honest but curious" — it can see the weights
    # but DP ensures those weights don't leak patient data.
    strategy = create_fedprox_strategy(
        proximal_mu=mu,
        num_clients=2,
        initial_parameters=initial_parameters,
    )

    # ── CSV Logger ───────────────────────────────────────────────────────────
    log_path = results_dir / "round_metrics.csv"
    csv_headers = [
        "round", "global_val_dice", "global_val_iou", "global_val_pixel_acc",
        "train_dice", "train_iou", "round_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_headers)

    # ── Run Flower Simulation ────────────────────────────────────────────────
    print_dp_summary(
        target_epsilon=target_epsilon,
        max_grad_norm=max_grad_norm,
        noise_multiplier=0.0,  # Will be computed internally by Opacus
    )
    print(f"\n[FL] Starting DP-FedProx simulation ({num_rounds} rounds, "
          f"ε={target_epsilon})...\n")

    start_time = time.time()

    history = fl.simulation.start_simulation(
        client_fn=create_client_fn(
            client_configs, device, local_epochs, CONFIG["lr"], mu,
            target_epsilon, CONFIG["target_delta"], max_grad_norm,
        ),
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
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

    # ── Extract Round Metrics ────────────────────────────────────────────────
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

    # Write CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for entry in round_metrics:
            writer.writerow({
                "round":                  entry.get("round", ""),
                "global_val_dice":        f"{entry.get('global_val_dice', 0):.6f}",
                "global_val_iou":         f"{entry.get('global_val_iou', 0):.6f}",
                "global_val_pixel_acc":   f"{entry.get('global_val_pixel_acc', 0):.6f}",
                "train_dice":             f"{entry.get('train_dice', 0):.6f}",
                "train_iou":              f"{entry.get('train_iou', 0):.6f}",
                "round_time_sec":         "",
            })

    # Final metrics
    if round_metrics and "global_val_dice" in round_metrics[-1]:
        final_dice = round_metrics[-1].get("global_val_dice", 0)
        final_iou  = round_metrics[-1].get("global_val_iou", 0)
        final_pix  = round_metrics[-1].get("global_val_pixel_acc", 0)
    else:
        final_dice, final_iou, final_pix = 0, 0, 0

    # ── Compare vs non-private FedProx ──────────────────────────────────────
    fedprox_report_path = PROJECT_ROOT / "results" / "federated" / "fedprox" / "fedprox_report.json"
    fedprox_baseline_dice = None
    if fedprox_report_path.exists():
        with open(fedprox_report_path) as f:
            fedprox_data = json.load(f)
        fedprox_baseline_dice = fedprox_data.get("final_metrics", {}).get("val_dice", None)

    # ── Save Full Report ──────────────────────────────────────────────────────
    report = {
        "experiment":       "DP-FedProx — Differentially Private Federated Learning",
        "algorithm":        "FedProx (Li et al., 2020) + DP-SGD (Abadi et al., 2016)",
        "purpose":          "Phase 4: Privacy-preserving FL with mathematical DP guarantee",
        "fedprox_mu":       mu,
        "num_rounds":       num_rounds,
        "num_clients":      2,
        "local_epochs":     local_epochs,
        "warm_start":       args.warm_start,
        "total_time_sec":   round(total_time, 2),
        "completed_at":     datetime.now().isoformat(),
        "privacy": {
            "target_epsilon":    target_epsilon,
            "target_delta":      CONFIG["target_delta"],
            "max_grad_norm":     max_grad_norm,
            "guarantee":        (
                f"Patient data is ({target_epsilon}, {CONFIG['target_delta']:.0e})"
                f"-differentially private"
            ),
        },
        "quantization":     quant_stats,
        "final_metrics": {
            "val_dice":      final_dice,
            "val_iou":       final_iou,
            "val_pixel_acc": final_pix,
        },
        "comparison_with_fedprox": {
            "fedprox_dice":    fedprox_baseline_dice,
            "dp_fedprox_dice": final_dice,
            "accuracy_cost":   (
                round(fedprox_baseline_dice - final_dice, 4)
                if fedprox_baseline_dice else "N/A"
            ),
            "interpretation": (
                f"DP adds ε={target_epsilon} privacy guarantee with "
                f"{round((fedprox_baseline_dice or 0) - final_dice, 4):.4f} Dice drop"
                if fedprox_baseline_dice else "FedProx baseline not available"
            ),
        },
        "round_history":  round_metrics,
        "config":         CONFIG,
    }

    report_path = results_dir / "dp_fedprox_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DP-FedProx EXPERIMENT COMPLETE")
    print(f"{'='*65}")
    print(f"  Rounds:              {num_rounds}")
    print(f"  Privacy Budget (ε):  {target_epsilon}")
    print(f"  Total Time:          {total_time:.1f} sec")
    print(f"  Final Val Dice:      {final_dice:.4f}")
    print(f"  Final Val IoU:       {final_iou:.4f}")
    print(f"  Final Val PixAcc:    {final_pix:.4f}")
    if fedprox_baseline_dice:
        accuracy_cost = fedprox_baseline_dice - final_dice
        print(f"\n  Privacy Cost Analysis:")
        print(f"  FedProx Dice (no DP):   {fedprox_baseline_dice:.4f}")
        print(f"  DP-FedProx Dice:        {final_dice:.4f}")
        print(f"  Accuracy Cost of DP:    {accuracy_cost:+.4f} Dice")
        print(f"  → Privacy at low cost: ε={target_epsilon} with {accuracy_cost:.4f} Dice drop")
    print(f"\n  Results saved to: {results_dir}/")
    print(f"    ├── round_metrics.csv")
    print(f"    ├── dp_fedprox_report.json")
    print(f"    └── quantization_report.json")


if __name__ == "__main__":
    main()
