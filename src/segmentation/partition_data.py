# src/segmentation/partition_data.py
# FedMedSeg Phase 3 — Non-IID Data Partitioner
#
# PURPOSE:
#   Splits the existing 4,000-image training subset into 2 Non-IID client
#   partitions that simulate realistic hospital data distributions:
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │  Client A ("Specialist Hospital"):  75% Pneumonia, 25% Normal │
#   │  Client B ("General Clinic"):       25% Pneumonia, 75% Normal │
#   └──────────────────────────────────────────────────────────────┘
#
#   This Non-IID (Non-Independent and Identically Distributed) split
#   simulates the real world where different hospitals see different
#   patient populations. Specialist centres see many sick patients;
#   small clinics see mostly healthy patients.
#
# WHY NON-IID?
#   If data were IID (same distribution everywhere), federation is trivial.
#   The *scientific contribution* is showing that FedAvg/FedProx can handle
#   skewed data and STILL converge to a good global model.
#
# OUTPUTS (saved to data/rsna_pneumonia/subset/):
#   client_a_train.csv   — 2,000 rows (1,500 pneumonia + 500 normal)
#   client_b_train.csv   — 2,000 rows (500 pneumonia  + 1,500 normal)
#   partition_summary.json — metadata and class distribution stats
#
# The VALIDATION set is NOT partitioned — it stays global (shared).
# This ensures fair evaluation across all experiments.

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_SEED = 42

# Non-IID split ratios
# Client A: Specialist hospital — sees mostly sick patients
CLIENT_A_PNEUMONIA = 1500
CLIENT_A_NORMAL    = 500

# Client B: General clinic — sees mostly healthy patients
CLIENT_B_PNEUMONIA = 500
CLIENT_B_NORMAL    = 1500


def partition_non_iid(
    train_csv: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    client_a_pneumonia: int = CLIENT_A_PNEUMONIA,
    client_a_normal: int = CLIENT_A_NORMAL,
    client_b_pneumonia: int = CLIENT_B_PNEUMONIA,
    client_b_normal: int = CLIENT_B_NORMAL,
) -> tuple:
    """
    Partition the training dataset into 2 Non-IID client datasets.

    Strategy:
      1. Separate training data by class (Pneumonia vs Normal).
      2. Assign samples to clients with deliberate label skew.
      3. Shuffle each client's data independently.
      4. Save as separate CSV files.

    The total across both clients equals the full training set (4,000 images).

    Args:
        train_csv (Path): Path to train_subset.csv (4,000 samples).
        output_dir (Path): Directory to save client CSVs.
        seed (int): Random seed for reproducibility.
        client_a_pneumonia (int): Pneumonia samples for Client A.
        client_a_normal (int): Normal samples for Client A.
        client_b_pneumonia (int): Pneumonia samples for Client B.
        client_b_normal (int): Normal samples for Client B.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (client_a_df, client_b_df)
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  Non-IID Data Partitioning for Federated Learning")
    print("=" * 60)

    # ── Step 1: Load training data ────────────────────────────────────────────
    if not train_csv.exists():
        raise FileNotFoundError(
            f"\nTraining CSV not found: {train_csv}\n"
            "Run prepare_subset.py first to create the data subset."
        )

    df = pd.read_csv(train_csv)
    print(f"\n[1] Loaded training data: {len(df)} samples")
    print(f"    Pneumonia: {(df['label'] == 1).sum()}")
    print(f"    Normal:    {(df['label'] == 0).sum()}")

    # ── Step 2: Separate by class ─────────────────────────────────────────────
    pneumonia_ids = df[df["label"] == 1]["patientId"].tolist()
    normal_ids    = df[df["label"] == 0]["patientId"].tolist()

    # Shuffle before splitting
    random.shuffle(pneumonia_ids)
    random.shuffle(normal_ids)

    # ── Step 3: Validate counts ───────────────────────────────────────────────
    total_pneumonia_needed = client_a_pneumonia + client_b_pneumonia
    total_normal_needed    = client_a_normal + client_b_normal

    if total_pneumonia_needed > len(pneumonia_ids):
        raise ValueError(
            f"Need {total_pneumonia_needed} pneumonia samples but only "
            f"{len(pneumonia_ids)} available."
        )
    if total_normal_needed > len(normal_ids):
        raise ValueError(
            f"Need {total_normal_needed} normal samples but only "
            f"{len(normal_ids)} available."
        )

    print(f"\n[2] Partitioning into Non-IID splits:")
    print(f"    Client A: {client_a_pneumonia} Pneumonia + {client_a_normal} Normal "
          f"= {client_a_pneumonia + client_a_normal} total  (75% / 25%)")
    print(f"    Client B: {client_b_pneumonia} Pneumonia + {client_b_normal} Normal "
          f"= {client_b_pneumonia + client_b_normal} total  (25% / 75%)")

    # ── Step 4: Assign samples ────────────────────────────────────────────────
    # Client A gets first slice of pneumonia, Client B gets second slice
    a_pneu_ids = pneumonia_ids[:client_a_pneumonia]
    b_pneu_ids = pneumonia_ids[client_a_pneumonia:client_a_pneumonia + client_b_pneumonia]

    # Client A gets first slice of normal, Client B gets second slice
    a_norm_ids = normal_ids[:client_a_normal]
    b_norm_ids = normal_ids[client_a_normal:client_a_normal + client_b_normal]

    # ── Step 5: Build DataFrames ──────────────────────────────────────────────
    client_a_df = pd.DataFrame({
        "patientId": a_pneu_ids + a_norm_ids,
        "label":     [1] * len(a_pneu_ids) + [0] * len(a_norm_ids),
    }).sample(frac=1, random_state=seed).reset_index(drop=True)

    client_b_df = pd.DataFrame({
        "patientId": b_pneu_ids + b_norm_ids,
        "label":     [1] * len(b_pneu_ids) + [0] * len(b_norm_ids),
    }).sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Step 6: Verify no overlap ─────────────────────────────────────────────
    overlap = set(client_a_df["patientId"]) & set(client_b_df["patientId"])
    assert len(overlap) == 0, f"ERROR: {len(overlap)} patients appear in both clients!"
    print(f"\n[3] Overlap check: ✓ No shared patients between clients")

    total = len(client_a_df) + len(client_b_df)
    print(f"    Total samples: {total} (should be {len(df)})")

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    a_csv = output_dir / "client_a_train.csv"
    b_csv = output_dir / "client_b_train.csv"

    client_a_df.to_csv(a_csv, index=False)
    client_b_df.to_csv(b_csv, index=False)

    # Summary JSON
    summary = {
        "created_at":     datetime.now().isoformat(),
        "random_seed":    seed,
        "strategy":       "Label-skew Non-IID (75/25 split)",
        "source_csv":     str(train_csv),
        "num_clients":    2,
        "client_a": {
            "name":       "Specialist Hospital",
            "total":      len(client_a_df),
            "pneumonia":  int((client_a_df["label"] == 1).sum()),
            "normal":     int((client_a_df["label"] == 0).sum()),
            "pneumonia_pct": round(
                (client_a_df["label"] == 1).mean() * 100, 1
            ),
            "csv_path":   str(a_csv),
        },
        "client_b": {
            "name":       "General Clinic",
            "total":      len(client_b_df),
            "pneumonia":  int((client_b_df["label"] == 1).sum()),
            "normal":     int((client_b_df["label"] == 0).sum()),
            "pneumonia_pct": round(
                (client_b_df["label"] == 1).mean() * 100, 1
            ),
            "csv_path":   str(b_csv),
        },
    }
    with open(output_dir / "partition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[4] Saved outputs to: {output_dir}/")
    print(f"    ├── client_a_train.csv  ({len(client_a_df)} rows)")
    print(f"    ├── client_b_train.csv  ({len(client_b_df)} rows)")
    print(f"    └── partition_summary.json")
    print("\n✓ Non-IID partitioning complete!")

    return client_a_df, client_b_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    train_csv    = project_root / "data" / "rsna_pneumonia" / "subset" / "train_subset.csv"
    output_dir   = project_root / "data" / "rsna_pneumonia" / "subset"

    client_a, client_b = partition_non_iid(
        train_csv=train_csv,
        output_dir=output_dir,
    )

    print("\nClient A (first 5 rows):")
    print(client_a.head())
    print("\nClient B (first 5 rows):")
    print(client_b.head())
