"""
continuous_client.py
FedMedSeg — Continuous Federated Learning Client (Hospital Node)

Unlike start_client.py (which exits after one batch of rounds), this client
stays alive across multiple server sessions. After each session the server
closes the gRPC connection; this script waits for the cool-down and then
reconnects automatically, ready for the next session.

Optional --data-watch mode
--------------------------
When enabled the client re-scans its local CSV directory before every
reconnect attempt. Any NEW patient rows that have been appended to the CSV
since the last session will be included in the next session's DataLoader.
This simulates incremental data arrival at a real hospital.

Usage
-----
  python continuous_client.py \\
      --server 192.168.1.100:8080 \\
      --node-type client_a \\
      --device auto \\
      --local-epochs 1 \\
      --batch-size 16 \\
      --mu 0.01 \\
      --reconnect-retries -1 \\   # -1 = retry indefinitely
      --reconnect-delay 30 \\     # seconds between reconnect attempts
      --data-watch                # re-scan CSV before each session
"""

import argparse
import time
import sys
from pathlib import Path

import flwr as fl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet
from segmentation.dataset_rsna import RSNAPneumoniaDataset
from segmentation.fl_client import FedMedSegClient


# ─────────────────────────────────────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


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


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset / DataLoader builder — re-called each session in --data-watch mode
# ─────────────────────────────────────────────────────────────────────────────

def build_data_loaders(args) -> tuple:
    """
    Build train + val DataLoaders for this client node.

    Called once at startup, and again before every reconnect attempt when
    --data-watch is active (to pick up newly arrived patient records).

    Returns:
        (train_loader, val_loader, train_dataset_size)
    """
    rsna_root = str(PROJECT_ROOT / "data" / "rsna_pneumonia")
    train_csv = str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset"
                    / f"{args.node_type}_train.csv")
    val_csv   = str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset"
                    / "val_subset.csv")

    if not Path(train_csv).exists():
        print(f"[Client] ❌  Dataset not found: {train_csv}")
        sys.exit(1)

    print(f"\n[Client] Loading dataset from: {train_csv}")
    train_ds = RSNAPneumoniaDataset(
        rsna_root=rsna_root,
        subset_csv=train_csv,
        img_transform=get_train_transform(),
        augment=True,
    )
    val_ds = RSNAPneumoniaDataset(
        rsna_root=rsna_root,
        subset_csv=val_csv,
        img_transform=get_val_transform(),
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
#  Main reconnect loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("FedMedSeg Continuous Hospital Node (Client)")
    parser.add_argument("--server",            type=str, required=True,
                        help="Server address e.g. 192.168.1.100:8080")
    parser.add_argument("--node-type",         type=str,
                        choices=["client_a", "client_b"], required=True,
                        help="Which data partition to use")
    parser.add_argument("--device",            type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-epochs",      type=int, default=1)
    parser.add_argument("--batch-size",        type=int, default=16)
    parser.add_argument("--mu",                type=float, default=0.01,
                        help="FedProx proximal coefficient (0.0 = FedAvg)")
    # DP
    parser.add_argument("--use-dp",            action="store_true",
                        help="Enable Differential Privacy")
    parser.add_argument("--epsilon",           type=float, default=8.0)
    parser.add_argument("--max-grad-norm",     type=float, default=1.0)
    # Continuous-pipeline specific
    parser.add_argument("--reconnect-retries", type=int, default=-1,
                        help="Number of reconnect attempts per session gap. "
                             "-1 = retry indefinitely until server comes back up")
    parser.add_argument("--reconnect-delay",   type=int, default=30,
                        help="Seconds to wait between reconnect attempts")
    parser.add_argument("--data-watch",        action="store_true",
                        help="Re-scan the local CSV before each session to pick "
                             "up newly arrived patient records")

    args = parser.parse_args()

    # ── Device ─────────────────────────────────────────────────────────────
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("\n" + "=" * 65)
    print(f"  FedMedSeg — Continuous Hospital Node  ({args.node_type})")
    print(f"  Server       : {args.server}")
    print(f"  Device       : {device}")
    print(f"  DP Enabled   : {args.use_dp}")
    print(f"  Data-Watch   : {args.data_watch}")
    strategy_label = f"FedProx (mu={args.mu})" if args.mu > 0 else "FedAvg"
    print(f"  Strategy     : {strategy_label}")
    print(f"  Reconnect    : {'∞' if args.reconnect_retries == -1 else args.reconnect_retries}"
          f" retries / {args.reconnect_delay}s delay")
    print("=" * 65)

    session_id = 0
    pipeline_start = time.time()

    # ── Initial data load ──────────────────────────────────────────────────
    train_loader, val_loader = build_data_loaders(args)

    try:
        while True:
            session_id += 1

            # ── (Re)build data loaders if data-watch is on ─────────────────
            if session_id > 1 and args.data_watch:
                print(f"\n[Client] --data-watch: refreshing dataset for Session {session_id}...")
                train_loader, val_loader = build_data_loaders(args)

            # ── (Re)initialise model for each session ──────────────────────
            # We always create a fresh PyTorch model object; the server will
            # push the aggregated global weights via set_parameters() at the
            # very start of the first round of every session, so the local
            # random initialisation here is immediately overwritten.
            print(f"\n[Client] Initialising model for Session {session_id}...")
            model = MobileNetV2UNet(pretrained=False, freeze_encoder=False).to(device)

            client = FedMedSegClient(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                client_name=args.node_type,
                local_epochs=args.local_epochs,
                lr=1e-4,
                mu=args.mu,
                use_dp=args.use_dp,
                target_epsilon=args.epsilon,
                max_grad_norm=args.max_grad_norm,
            )

            # ── Connect + train ────────────────────────────────────────────
            attempt = 0
            session_done = False
            session_start = time.time()

            while not session_done:
                attempt += 1
                max_label = (str(args.reconnect_retries)
                             if args.reconnect_retries != -1
                             else "∞")
                print(f"\n[Client] Session {session_id} — "
                      f"Connect attempt {attempt}/{max_label}  →  {args.server}")

                try:
                    fl.client.start_numpy_client(
                        server_address=args.server,
                        client=client,
                    )
                    # Reached here = server closed the connection cleanly
                    # after the final round (expected end-of-session).
                    session_done = True

                except Exception as e:
                    err_msg = str(e).lower()

                    # Clean server shutdown after last round — treat as success
                    if any(sig in err_msg for sig in [
                        "stopiteration", "channel is closed", "server closed",
                        "failed to connect",
                    ]):
                        # "failed to connect" means the server is in cool-down
                        # between sessions; we should retry after a delay.
                        if "failed to connect" in err_msg or "connection refused" in err_msg:
                            # Check retry budget
                            if args.reconnect_retries != -1 and attempt >= args.reconnect_retries:
                                print(f"[Client] Exhausted {args.reconnect_retries} reconnect "
                                      f"attempts. Stopping client.")
                                return
                            print(f"[Client] Server not yet available (Session {session_id + 1} "
                                  f"cool-down). Retrying in {args.reconnect_delay}s...")
                            time.sleep(args.reconnect_delay)
                        else:
                            # Any other clean close = end of session
                            session_done = True
                    else:
                        # Unexpected error — log it and retry
                        print(f"[Client] Unexpected error: {e}")
                        if args.reconnect_retries != -1 and attempt >= args.reconnect_retries:
                            print(f"[Client] Exhausted retries. Stopping.")
                            return
                        print(f"[Client] Retrying in {args.reconnect_delay}s...")
                        time.sleep(args.reconnect_delay)

            # ── Session complete ────────────────────────────────────────────
            elapsed = time.time() - session_start
            total   = time.time() - pipeline_start
            print("\n" + "=" * 65)
            print(f"  ✅  SESSION {session_id:03d} COMPLETE  ({args.node_type})")
            print(f"  Session time : {elapsed / 60:.1f} min ({elapsed:.0f} sec)")
            print(f"  Pipeline uptime: {total / 60:.1f} min")
            print("=" * 65)

            # ── Wait a moment before probing for the next session ──────────
            # The server takes some time to restart between sessions.
            # We wait reconnect_delay seconds before the first probe.
            print(f"\n[Client] Waiting {args.reconnect_delay}s before probing "
                  f"for Session {session_id + 1}...")
            time.sleep(args.reconnect_delay)

    except KeyboardInterrupt:
        total = time.time() - pipeline_start
        print(f"\n\n[Client] Ctrl+C received — stopping after Session {session_id}.")
        print(f"[Client] Total pipeline uptime: {total / 60:.1f} min\n")


if __name__ == "__main__":
    main()
