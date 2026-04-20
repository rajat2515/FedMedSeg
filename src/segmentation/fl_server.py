# src/segmentation/fl_server.py
# FedMedSeg Phase 3 — Flower Federated Learning Server
#
# The server is the AGGREGATOR in federated learning. It:
#   1. Distributes the global model to all clients
#   2. Collects updated weights after local training
#   3. Aggregates weights using FedAvg (weighted average by dataset size)
#   4. Evaluates the new global model
#   5. Repeats for N rounds
#
# AGGREGATION FORMULA (FedAvg):
#   w_global = Σ (n_k / n_total) * w_k
#   Where:
#     n_k     = number of training samples on client k
#     n_total = total samples across all clients
#     w_k     = model weights from client k after local training
#
# This module provides helper functions used by the runner scripts
# (run_fedavg.py, run_fedprox.py) to configure the Flower simulation.

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC AGGREGATION — Combine Per-Client Metrics into Global Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute a weighted average of per-client metrics.

    This is called by the Flower server after each evaluation round to
    produce a single set of global metrics.

    The weighting is proportional to each client's dataset size:
      metric_global = Σ (n_k / n_total) * metric_k

    Args:
        metrics: List of (num_examples, metrics_dict) tuples, one per client.

    Returns:
        Metrics: Weighted-average dict {val_dice, val_iou, val_pixel_acc}.
    """
    total_examples = sum(n for n, _ in metrics)

    if total_examples == 0:
        return {"val_dice": 0.0, "val_iou": 0.0, "val_pixel_acc": 0.0}

    # Weighted sum for each metric key
    aggregated = {}
    for key in ["val_dice", "val_iou", "val_pixel_acc"]:
        weighted_sum = sum(
            n * m.get(key, 0.0) for n, m in metrics
        )
        aggregated[key] = weighted_sum / total_examples

    return aggregated


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate training metrics from fit() across all clients.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from fit().

    Returns:
        Metrics: Weighted-average training metrics.
    """
    total_examples = sum(n for n, _ in metrics)

    if total_examples == 0:
        return {}

    aggregated = {}
    for key in ["train_loss", "train_dice", "train_iou", "train_pixel_acc"]:
        weighted_sum = sum(
            n * m.get(key, 0.0) for n, m in metrics
        )
        aggregated[key] = weighted_sum / total_examples

    return aggregated


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY FACTORY — Create Flower Strategies
# ═══════════════════════════════════════════════════════════════════════════════

def create_fedavg_strategy(
    num_clients: int = 2,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    initial_parameters: Optional[fl.common.Parameters] = None,
) -> fl.server.strategy.FedAvg:
    """
    Create a FedAvg strategy for the Flower server.

    FedAvg (Federated Averaging) is the baseline FL algorithm:
      - Each client trains locally for E epochs
      - Server averages weights proportional to dataset size
      - No regularization — clients are free to diverge

    Args:
        num_clients (int): Total number of participating clients.
        fraction_fit (float): Fraction of clients to train per round.
        fraction_evaluate (float): Fraction of clients to evaluate per round.
        min_fit_clients (int): Minimum clients required for training.
        min_evaluate_clients (int): Minimum clients for evaluation.
        min_available_clients (int): Minimum clients that must be connected.
        initial_parameters: Optional initial model parameters.

    Returns:
        fl.server.strategy.FedAvg: Configured FedAvg strategy.
    """
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )


def create_fedprox_strategy(
    proximal_mu: float = 0.01,
    num_clients: int = 2,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    initial_parameters: Optional[fl.common.Parameters] = None,
) -> fl.server.strategy.FedProx:
    """
    Create a FedProx strategy for the Flower server.

    FedProx extends FedAvg by adding a proximal term on the CLIENT side:
      L_prox = L_task + (μ/2) * ||w_local - w_global||²

    NOTE: The proximal term is implemented in fl_client.py's train_local().
    The server-side FedProx strategy in Flower handles the configuration
    passing. The actual proximal penalty is computed during local training.

    Args:
        proximal_mu (float): Proximal coefficient μ. Higher = more conservative.
            Typical values: 0.001 to 0.1.
            μ = 0 reduces to FedAvg.
        Other args: Same as create_fedavg_strategy.

    Returns:
        fl.server.strategy.FedProx: Configured FedProx strategy.
    """
    return fl.server.strategy.FedProx(
        proximal_mu=proximal_mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )
