"""
continuous_server.py
FedMedSeg — Continuous Federated Learning Server

Runs the Flower server in a session loop:
  Session 1  →  N rounds  →  save checkpoint  →  wait interval
  Session 2  →  N rounds  →  save checkpoint  →  wait interval
  ...

Each new session warm-starts from the BEST global model checkpoint of all
prior sessions so learning is cumulative, not from scratch.

Usage
-----
  python continuous_server.py \\
      --host 0.0.0.0 \\
      --port 8080 \\
      --rounds 20 \\
      --clients 2 \\
      --strategy fedprox \\
      --mu 0.01 \\
      --sessions -1 \\          # -1 = run indefinitely until Ctrl+C
      --interval 3600 \\        # seconds between sessions (0 = back-to-back)
      --checkpoint-dir checkpoints/

Checkpoints saved
-----------------
  checkpoints/
  ├── best_global_model.pth        ← rolling best (by Val Dice)
  ├── session_001_model.pth
  ├── session_002_model.pth
  └── ...
  pipeline_log.json                ← appended after every session
"""

import argparse
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import flwr as fl

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet
from segmentation.fl_client import get_parameters
from segmentation.fl_server import (
    create_fedavg_strategy,
    create_fedprox_strategy,
    save_global_model,
    load_global_model_parameters,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_strategy(args, initial_parameters):
    """Instantiate the Flower strategy with the given initial parameters."""
    common = dict(
        num_clients=args.clients,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=initial_parameters,
    )
    if args.strategy == "fedprox":
        return create_fedprox_strategy(proximal_mu=args.mu, **common)
    return create_fedavg_strategy(**common)


def _extract_metrics(history, num_rounds: int):
    """Pull the last-round distributed metrics from a Flower History object."""
    result = {"val_dice": None, "val_iou": None, "val_pixel_acc": None}
    if not (history and history.metrics_distributed):
        return result
    metrics = history.metrics_distributed
    for key in result:
        vals = metrics.get(key, [])
        if vals:
            result[key] = vals[-1][1]
    return result


def _print_session_banner(session_id: int, args):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 65)
    print(f"  🚀  SESSION {session_id:03d}  —  {ts}")
    print(f"  Strategy  : {args.strategy.upper()}  (mu={args.mu if args.strategy == 'fedprox' else 'N/A'})")
    print(f"  Rounds    : {args.rounds}")
    print(f"  Clients   : {args.clients}")
    print("=" * 65)


def _print_session_summary(session_id, elapsed, metrics, is_best):
    best_tag = "  ⭐  NEW BEST" if is_best else ""
    print("\n" + "=" * 65)
    print(f"  ✅  SESSION {session_id:03d} COMPLETE{best_tag}")
    print(f"  Time       : {elapsed / 60:.1f} min ({elapsed:.0f} sec)")
    for k, v in metrics.items():
        if v is not None:
            label = k.replace("val_", "").replace("_", " ").title()
            print(f"  {label:<14}: {v:.4f}")
    print("=" * 65)


def _append_pipeline_log(log_path: Path, entry: dict):
    """Append one session entry to pipeline_log.json."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                log = json.load(f)
        except json.JSONDecodeError:
            pass
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def _print_pipeline_summary(log_path: Path):
    """Print a compact table of all session results so far."""
    if not log_path.exists():
        return
    try:
        with open(log_path) as f:
            log = json.load(f)
    except Exception:
        return
    print("\n  📊  Pipeline Summary Across All Sessions:")
    print(f"  {'Sess':>5}  {'Date':>19}  {'Dice':>6}  {'IoU':>6}  {'PixAcc':>7}")
    print("  " + "-" * 50)
    for e in log:
        dice   = f"{e['val_dice']:.4f}"   if e.get("val_dice")   else "  N/A"
        iou    = f"{e['val_iou']:.4f}"    if e.get("val_iou")    else "  N/A"
        pixacc = f"{e['val_pixel_acc']:.4f}" if e.get("val_pixel_acc") else "  N/A"
        print(f"  {e['session']:>5}  {e['timestamp']:>19}  {dice:>6}  {iou:>6}  {pixacc:>7}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main session loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("FedMedSeg Continuous Training Server")
    parser.add_argument("--host",           type=str,   default="0.0.0.0")
    parser.add_argument("--port",           type=int,   default=8080)
    parser.add_argument("--rounds",         type=int,   default=20,
                        help="FL rounds per session")
    parser.add_argument("--clients",        type=int,   default=2,
                        help="Number of hospital nodes to wait for")
    parser.add_argument("--strategy",       type=str,   choices=["fedavg", "fedprox"],
                        default="fedprox")
    parser.add_argument("--mu",             type=float, default=0.01,
                        help="FedProx proximal coefficient")
    parser.add_argument("--sessions",       type=int,   default=-1,
                        help="Number of sessions to run. -1 = infinite (until Ctrl+C)")
    parser.add_argument("--interval",       type=int,   default=3600,
                        help="Seconds to wait between sessions. 0 = back-to-back")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path      = PROJECT_ROOT / "pipeline_log.json"
    best_ckpt     = checkpoint_dir / "best_global_model.pth"

    print("\n" + "=" * 65)
    print("  FedMedSeg — Continuous Federated Learning Pipeline")
    print(f"  Address      : {args.host}:{args.port}")
    print(f"  Sessions     : {'∞  (until Ctrl+C)' if args.sessions == -1 else args.sessions}")
    print(f"  Rounds/session: {args.rounds}")
    print(f"  Interval     : {args.interval}s between sessions")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 65)

    # ── Persistent model shell (used only for key extraction / saving) ──────
    model_shell = MobileNetV2UNet(pretrained=False, freeze_encoder=False)

    best_dice   = -1.0
    session_id  = 0

    try:
        while True:
            session_id += 1
            if args.sessions != -1 and session_id > args.sessions:
                print(f"\n[Pipeline] All {args.sessions} sessions complete. Shutting down.")
                break

            _print_session_banner(session_id, args)

            # ── Warm-start: load best checkpoint if it exists ──────────────
            if best_ckpt.exists():
                print(f"[Pipeline] Warm-starting session {session_id} from: {best_ckpt.name}")
                initial_parameters = load_global_model_parameters(best_ckpt, model_shell)
            else:
                print(f"[Pipeline] No checkpoint found. Initialising fresh model weights.")
                init_model = MobileNetV2UNet(pretrained=True, freeze_encoder=False)
                initial_parameters = fl.common.ndarrays_to_parameters(
                    get_parameters(init_model)
                )
                del init_model

            strategy = _build_strategy(args, initial_parameters)

            print(f"\n[Server] Waiting for {args.clients} client(s) to connect...\n")
            session_start = time.time()

            history = fl.server.start_server(
                server_address=f"{args.host}:{args.port}",
                config=fl.server.ServerConfig(num_rounds=args.rounds),
                strategy=strategy,
            )

            elapsed = time.time() - session_start
            metrics = _extract_metrics(history, args.rounds)

            # ── Save per-session checkpoint ────────────────────────────────
            # We need the final aggregated parameters from the history.
            # Flower stores them in history.parameters_distributed or via
            # the strategy's aggregated weights. We extract from the strategy.
            session_ckpt = checkpoint_dir / f"session_{session_id:03d}_model.pth"

            # Reconstruct final weights from strategy (parameters are passed
            # back as initial_parameters for the *next* session via the strategy
            # object which holds the last aggregated result).
            final_parameters = strategy.initial_parameters
            if final_parameters is not None:
                save_global_model(final_parameters, model_shell, session_ckpt)
                print(f"[Pipeline] Saved session checkpoint → {session_ckpt.name}")

                # ── Update best checkpoint ─────────────────────────────────
                current_dice = metrics.get("val_dice") or -1.0
                is_best = current_dice > best_dice
                if is_best:
                    best_dice = current_dice
                    save_global_model(final_parameters, model_shell, best_ckpt)
                    print(f"[Pipeline] ⭐  New best Val Dice = {best_dice:.4f} → {best_ckpt.name}")
            else:
                is_best = False
                print("[Pipeline] Warning: no aggregated parameters returned this session.")

            _print_session_summary(session_id, elapsed, metrics, is_best)

            # ── Log to pipeline_log.json ───────────────────────────────────
            log_entry = {
                "session":       session_id,
                "timestamp":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "rounds":        args.rounds,
                "total_time_sec": round(elapsed, 1),
                **metrics,
                "is_best":       is_best,
            }
            _append_pipeline_log(log_path, log_entry)
            _print_pipeline_summary(log_path)

            # ── Check if we are done ───────────────────────────────────────
            if args.sessions != -1 and session_id >= args.sessions:
                print(f"\n[Pipeline] All {args.sessions} sessions complete. Shutting down.")
                break

            # ── Cool-down before next session ──────────────────────────────
            if args.interval > 0:
                next_ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n[Pipeline] Cool-down: waiting {args.interval}s before Session "
                      f"{session_id + 1}. Next start ≈ {next_ts}")
                print("           (Press Ctrl+C at any time to stop the pipeline.)\n")
                time.sleep(args.interval)
            else:
                print(f"\n[Pipeline] Starting Session {session_id + 1} immediately...\n")

    except KeyboardInterrupt:
        print("\n\n[Pipeline] Ctrl+C received — stopping after current session.")
        _print_pipeline_summary(log_path)
        print(f"\n[Pipeline] Best global model saved at: {best_ckpt}")
        print(f"[Pipeline] Full log at: {log_path}")
        print("\n[Pipeline] Pipeline stopped. Goodbye.\n")


if __name__ == "__main__":
    main()
