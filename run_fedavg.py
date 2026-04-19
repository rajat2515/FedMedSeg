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
    4. Server aggregates:  w_{r+1} = S (n_k / n_total) x w_k
       where n_k = number of training samples on client k

  The key insight: DATA NEVER LEAVES THE HOSPITAL.
  Only model weights are transmitted - preserving patient privacy.

ARCHITECTURE (No Ray):
  Uses Python multiprocessing to simulate federated training on one machine.
  - Main process  : Flower Server  (listens on 0.0.0.0:8080)
  - Child process 0: Flower Client A (connects to 127.0.0.1:8080)
  - Child process 1: Flower Client B (connects to 127.0.0.1:8080)
  This is identical to running on separate machines - just change the IP.

RUN (Windows):
    .\\venv312\\Scripts\\python run_fedavg.py

OPTIONAL FLAGS:
    --device cpu|cuda    (default: auto-detect)
    --rounds 20          (default: 20)
    --local-epochs 1     (default: 1)
    --warm-start         (initialize from model3c_best.pth)
"""

# -- Standard Library ----------------------------------------------------------
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# -- Third-Party ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import flwr as fl

# -- Project Imports -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

from segmentation.model_unet import MobileNetV2UNet
from segmentation.loss import DiceBCELoss
from segmentation.metrics import compute_all_metrics
from segmentation.dataset_rsna import RSNAPneumoniaDataset
from segmentation.fl_client import (
    FedMedSegClient, get_parameters, set_parameters,
    evaluate_local,
)
from segmentation.fl_server import create_fedavg_strategy


# =============================================================================
#  CONFIGURATION
# =============================================================================

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
    return torch.device("cpu")


# =============================================================================
#  MULTIPROCESSING CLIENT RUNNER
#  IMPORTANT: This function must be defined at the TOP LEVEL of the module
#  (not inside main()) so that Python's multiprocessing 'spawn' method on
#  Windows can pickle and import it correctly.
# =============================================================================

def run_client(cid: str, client_configs: dict, device_str: str, local_epochs: int, lr: float, lock):
    """
    Runs a single Flower client in its own child process.
    Connects to the server at 127.0.0.1:8080 (localhost).
    On a real multi-machine deployment, change this IP to the server's IP.
    """
    # Re-add src to path since child processes start fresh
    import sys, os
    from pathlib import Path
    _root = Path(__file__).resolve().parent
    sys.path.insert(0, str(_root / "src"))

    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    import flwr as fl
    from segmentation.model_unet import MobileNetV2UNet
    from segmentation.dataset_rsna import RSNAPneumoniaDataset
    from segmentation.fl_client import FedMedSegClient

    device = torch.device(device_str)
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

    client = FedMedSegClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        client_name=cfg["name"],
        local_epochs=local_epochs,
        lr=lr,
        mu=0.0,  # FedAvg — no proximal term
        lock=lock, # Pass the multiprocessing lock to prevent GPU thrashing
    )

    print(f"  [Client {cid}] {cfg['name']} connecting to server...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    print(f"  [Client {cid}] {cfg['name']} finished.")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FedAvg Experiment")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
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

    num_rounds   = args.rounds
    local_epochs = args.local_epochs

    print("\n" + "=" * 65)
    print("  FEDERATED AVERAGING (FedAvg) EXPERIMENT")
    print(f"  Device: {device}  |  Rounds: {num_rounds}  |  Local Epochs: {local_epochs}")
    print("  2 hospital clients collaborate by sharing weights, NOT data.")
    print("  Backend: Python multiprocessing (no Ray required)")
    print("=" * 65)

    # -- Verify CSVs -----------------------------------------------------------
    print("\n[Data] Verifying dataset CSVs exist...")
    for csv_path in [CONFIG["client_a_csv"], CONFIG["client_b_csv"], CONFIG["val_csv"]]:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"  [OK] {Path(csv_path).name}")

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

    # -- Initial Global Model --------------------------------------------------
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
            print(f"  [WARN] Warm start requested but {ckpt_path} not found. Using ImageNet weights.")

    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    # -- Strategy --------------------------------------------------------------
    strategy = create_fedavg_strategy(
        num_clients=2,
        initial_parameters=initial_parameters,
    )

    # -- CSV Logger ------------------------------------------------------------
    log_path = results_dir / "round_metrics.csv"
    csv_headers = [
        "round", "global_val_dice", "global_val_iou", "global_val_pixel_acc",
        "train_dice", "train_iou", "round_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_headers)

    # -- Launch Client Processes -----------------------------------------------
    print(f"\n[FL] Spawning 2 client processes...")
    processes = []
    
    # Create a single lock to share between all clients
    # This forces them to take turns using the GPU, preventing memory deadlocks!
    gpu_lock = mp.Lock()
    
    for cid in ["0", "1"]:
        p = mp.Process(
            target=run_client,
            args=(cid, client_configs, str(device), local_epochs, CONFIG["lr"], gpu_lock),
            daemon=True,  # auto-killed if main process crashes
        )
        p.start()
        processes.append(p)

    # Give clients time to load datasets before the server starts accepting
    print("[FL] Waiting for clients to initialize (5 seconds)...")
    time.sleep(5)

    # -- Start Flower Server (blocks until all rounds complete) ----------------
    print(f"[FL] Starting FedAvg server ({num_rounds} rounds, 2 clients)...\n")
    start_time = time.time()

    try:
        history = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        print("\n[FL] Server interrupted by user.")
        history = None
    finally:
        # Clean up client processes regardless of outcome
        print("\n[FL] Cleaning up client processes...")
        for p in processes:
            p.terminate()
            p.join(timeout=10)

    total_time = time.time() - start_time

    if history is None:
        print("Training was interrupted. No results to save.")
        return

    # -- Extract & Save Round Metrics ------------------------------------------
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

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for entry in round_metrics:
            writer.writerow({
                "round":               entry.get("round", ""),
                "global_val_dice":     f"{entry.get('global_val_dice', 0):.6f}",
                "global_val_iou":      f"{entry.get('global_val_iou', 0):.6f}",
                "global_val_pixel_acc":f"{entry.get('global_val_pixel_acc', 0):.6f}",
                "train_dice":          f"{entry.get('train_dice', 0):.6f}",
                "train_iou":           f"{entry.get('train_iou', 0):.6f}",
                "round_time_sec":      "",
            })

    final_dice = round_metrics[-1].get("global_val_dice", 0) if round_metrics else 0
    final_iou  = round_metrics[-1].get("global_val_iou",  0) if round_metrics else 0
    final_pix  = round_metrics[-1].get("global_val_pixel_acc", 0) if round_metrics else 0

    # -- Save Report -----------------------------------------------------------
    report = {
        "experiment":     "Federated Averaging (FedAvg)",
        "algorithm":      "FedAvg -- McMahan et al., 2017",
        "backend":        "Python multiprocessing (no Ray)",
        "num_rounds":     num_rounds,
        "num_clients":    2,
        "local_epochs":   local_epochs,
        "device":         str(device),
        "warm_start":     args.warm_start,
        "total_time_sec": round(total_time, 2),
        "completed_at":   datetime.now().isoformat(),
        "final_metrics":  {"val_dice": final_dice, "val_iou": final_iou, "val_pixel_acc": final_pix},
        "round_history":  round_metrics,
        "config":         CONFIG,
    }

    with open(results_dir / "fedavg_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  FedAvg EXPERIMENT COMPLETE")
    print(f"  Rounds: {num_rounds}  |  Total Time: {total_time:.1f} sec")
    print(f"  Final Val Dice:   {final_dice:.4f}")
    print(f"  Final Val IoU:    {final_iou:.4f}")
    print(f"  Final Val PixAcc: {final_pix:.4f}")
    print(f"  Results saved to: {results_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    # Required on Windows for multiprocessing 'spawn' method
    mp.freeze_support()
    main()
