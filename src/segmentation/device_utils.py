# src/segmentation/device_utils.py
# Unified device selection for multi-platform training.
#
# Supports:
#   --device cuda   → NVIDIA GPU
#   --device mps    → Apple Silicon GPU
#   --device cpu    → CPU only
#   --device auto   → Auto-detect best available (default)
#
# Usage:
#   from segmentation.device_utils import add_device_arg, get_device, safe_empty_cache
#
#   parser = argparse.ArgumentParser()
#   add_device_arg(parser)                    # adds --device flag
#   args = parser.parse_args()
#   device = get_device(args.device)          # returns torch.device
#   ...
#   safe_empty_cache(device)                  # frees GPU memory safely

import torch


DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]


def add_device_arg(parser):
    """Add a --device argument to an argparse parser."""
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=DEVICE_CHOICES,
        help=(
            "Device to train on. "
            "'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' for CPU. "
            "'auto' picks the best available (default: auto)."
        ),
    )


def get_device(choice: str = "auto") -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        choice: One of 'auto', 'cuda', 'mps', 'cpu'.

    Returns:
        torch.device for the requested (or best-available) backend.
    """
    choice = choice.lower().strip()

    if choice == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif choice == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available — falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("WARNING: MPS requested but not available — falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _print_device_info(device)
    return device


def safe_empty_cache(device: torch.device):
    """Free GPU memory regardless of backend. Safe to call on CPU (no-op)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # torch.mps.empty_cache() available in PyTorch ≥ 2.1
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _print_device_info(device: torch.device):
    """Pretty-print which device was selected."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"Device: {device}  ({name})")
    elif device.type == "mps":
        print(f"Device: {device}  (Apple Silicon GPU)")
    else:
        print(f"Device: {device}")
