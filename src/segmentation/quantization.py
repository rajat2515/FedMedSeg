# src/segmentation/quantization.py
# FedMedSeg Phase 4 — Model Quantization Module
#
# PURPOSE:
#   Reduce the size of model weights BEFORE transmitting them from hospital
#   (client) to the aggregation server. Smaller payloads = faster federation.
#
# WHY QUANTIZATION MATTERS FOR FEDERATED LEARNING:
#   In each communication round, every hospital sends its full model weights
#   to the server. Our MobileNetV2-UNet has ~3.2 million parameters, each
#   stored as a 32-bit float (4 bytes).
#
#   Without quantization:  3.2M × 4 bytes = ~12.8 MB per round
#   With int8 quantization: 3.2M × 1 byte  = ~3.2 MB per round
#   Saving: ~75% reduction in communication bandwidth.
#
#   At 20 rounds with 2 clients, unquantized = 20 × 2 × 12.8MB = 512MB total.
#   With quantization:                       = 20 × 2 ×  3.2MB = 128MB total.
#
# WHAT TYPE OF QUANTIZATION DO WE USE?
#   Post-Training Dynamic Quantization (PTDQ):
#   - "Post-Training": No retraining required. We quantize the final trained
#     model weights on-the-fly before sending.
#   - "Dynamic": Only LINEAR layers are quantized to int8. Activations remain
#     float32 during computation. This gives a good accuracy/size trade-off.
#   - "int8": Each weight value is represented as an 8-bit integer instead of
#     a 32-bit float. Values are scaled back to float during computation.
#
# WHAT WE DO NOT USE:
#   Quantization-Aware Training (QAT) — requires retraining with fake quantization
#   nodes. Too complex for our purposes and marginal accuracy gain.
#
# NOTE ON CONVOLUTIONAL LAYERS:
#   PyTorch's dynamic quantization only supports Linear and LSTM layers.
#   Convolutional layers (Conv2d) are NOT quantized by dynamic quantization.
#   This is fine — the U-Net's bottleneck linear layers contribute significantly
#   to the model size, and Conv2d layers are less amenable to int8 compression.
#
# REFERENCES:
#   PyTorch Quantization Docs:
#   https://pytorch.org/docs/stable/quantization.html

import copy
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL SIZE MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_model_size_mb(model: nn.Module) -> float:
    """
    Measure the size of a PyTorch model's state_dict in megabytes.

    Uses an in-memory buffer to simulate serialization, giving an accurate
    estimate of the network payload size.

    Args:
        model (nn.Module): The PyTorch model to measure.

    Returns:
        float: Model state_dict size in megabytes.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return size_bytes / (1024 * 1024)


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        dict: {"total": int, "trainable": int, "frozen": int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  QUANTIZATION
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply Post-Training Dynamic Quantization to a model.

    This converts Linear layer weights from float32 → int8, reducing
    the model state_dict size by approximately 60–75%.

    The quantized model is returned as a NEW object — the original model
    is NOT modified (safe for continued training of the original).

    Args:
        model (nn.Module): Trained PyTorch model to quantize.

    Returns:
        nn.Module: A quantized copy of the model (CPU only).

    Note:
        Dynamic quantization only runs on CPU. Do NOT move the returned
        quantized model to CUDA — it will raise an error.
    """
    # Deep copy to avoid modifying the original model
    model_copy = copy.deepcopy(model).cpu()

    # Apply dynamic quantization to all Linear layers
    quantized = torch.quantization.quantize_dynamic(
        model_copy,
        qconfig_spec={nn.Linear},   # Only quantize Linear layers
        dtype=torch.qint8,          # Target: 8-bit signed integer
    )

    return quantized


def measure_compression(
    original_model: nn.Module,
    quantized_model: nn.Module,
) -> Dict[str, float]:
    """
    Measure and report size reduction from quantization.

    Args:
        original_model (nn.Module): The full float32 model.
        quantized_model (nn.Module): The quantized model.

    Returns:
        dict: Size measurements and reduction percentage.
    """
    original_mb = get_model_size_mb(original_model)
    quantized_mb = get_model_size_mb(quantized_model)
    reduction_pct = (1.0 - quantized_mb / original_mb) * 100.0

    stats = {
        "original_size_mb": round(original_mb, 2),
        "quantized_size_mb": round(quantized_mb, 2),
        "reduction_pct": round(reduction_pct, 1),
        "compression_ratio": round(original_mb / quantized_mb, 2),
    }

    return stats


def print_quantization_report(stats: Dict[str, float]):
    """Print a formatted quantization summary."""
    print("\n" + "─" * 55)
    print("  MODEL QUANTIZATION REPORT (float32 → int8)")
    print("─" * 55)
    print(f"  Original Size:     {stats['original_size_mb']:.2f} MB  (float32)")
    print(f"  Quantized Size:    {stats['quantized_size_mb']:.2f} MB  (int8 Linear)")
    print(f"  Size Reduction:    {stats['reduction_pct']:.1f}%")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}×")
    print(f"")
    print(f"  Per-round bandwidth savings:")
    saved = stats['original_size_mb'] - stats['quantized_size_mb']
    print(f"    {saved:.2f} MB × 2 clients × 20 rounds = "
          f"{saved * 2 * 20:.1f} MB total saved")
    print("─" * 55 + "\n")
