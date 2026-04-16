# src/segmentation/prepare_subset.py
# FedMedSeg Phase 2 — RSNA 5,000-image Balanced Subset Extractor
#
# PURPOSE:
#   The full RSNA Pneumonia Detection dataset contains ~26,000+ images.
#   Training on the full dataset would take many hours.
#   For Phase 2 demonstration, we extract a balanced subset of:
#       • 2,500 Pneumonia patients  (Target = 1, have bounding boxes)
#       • 2,500 Normal patients     (Target = 0, no bounding boxes)
#   Total: 5,000 images — enough to prove the U-Net works, fast to train.
#
# SPLIT STRATEGY:
#   80% Training (4,000 images): Used to teach the model.
#   20% Validation (1,000 images): Used to check performance during training.
#   The split is STRATIFIED — same class ratio in train and validation.
#
# OUTPUTS (saved to data/rsna_pneumonia/subset/):
#   train_subset.csv   — columns: patientId, label
#   val_subset.csv     — columns: patientId, label
#   subset_summary.txt — class counts, split sizes, random seed used

import pandas as pd
import numpy as np
from pathlib import Path
import random
import json
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

RSNA_ROOT      = Path("data/rsna_pneumonia")
LABELS_CSV     = RSNA_ROOT / "stage_2_train_labels.csv"
OUTPUT_DIR     = RSNA_ROOT / "subset"

PNEUMONIA_COUNT = 2500   # Number of pneumonia patients
NORMAL_COUNT    = 2500   # Number of normal patients
TRAIN_RATIO     = 0.80   # 80% training, 20% validation
RANDOM_SEED     = 42     # For reproducibility — ALWAYS document your seed!


def prepare_balanced_subset(
    labels_csv: Path = LABELS_CSV,
    output_dir: Path = OUTPUT_DIR,
    pneumonia_count: int = PNEUMONIA_COUNT,
    normal_count: int = NORMAL_COUNT,
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
    images_dir: Path = None,
) -> tuple:
    """
    Extract a balanced subset from the RSNA Pneumonia Detection dataset.

    Steps:
      1. Load the labels CSV.
      2. Identify unique Pneumonia patients (Target=1, have bounding boxes).
      3. Identify unique Normal patients (Target=0).
      4. Randomly sample the required counts from each group.
      5. Optionally verify that the corresponding DICOM files exist.
      6. Split into train/val with stratification.
      7. Save two CSVs and a summary text file.

    Args:
        labels_csv     (Path): Path to stage_2_train_labels.csv.
        output_dir     (Path): Directory to save subset CSVs.
        pneumonia_count (int): Number of pneumonia patients to sample.
        normal_count   (int): Number of normal patients to sample.
        train_ratio    (float): Fraction used for training.
        seed           (int): Random seed for reproducibility.
        images_dir     (Path): Optional — if provided, verifies .dcm files exist.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  RSNA Subset Preparation")
    print("=" * 60)

    # ── Step 1: Load full labels ───────────────────────────────────────────────
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"\nLabels CSV not found: {labels_csv}\n"
            "Please download the RSNA Pneumonia Detection Challenge dataset from:\n"
            "  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data\n"
            "and extract it to:  data/rsna_pneumonia/"
        )

    df = pd.read_csv(labels_csv)
    print(f"\n[1] Loaded labels CSV: {len(df)} rows")
    print(f"    Columns: {list(df.columns)}")

    # ── Step 2: Get unique patient IDs per class ──────────────────────────────
    # Note: A pneumonia patient can appear in MULTIPLE rows (one per bounding box).
    # We work at the PATIENT level, not the ROW level.
    pneumonia_patients = df[df["Target"] == 1]["patientId"].unique().tolist()
    normal_patients    = df[df["Target"] == 0]["patientId"].unique().tolist()

    print(f"\n[2] Unique Pneumonia patients in full dataset: {len(pneumonia_patients)}")
    print(f"    Unique Normal    patients in full dataset: {len(normal_patients)}")

    # ── Step 3: Validate requested counts ─────────────────────────────────────
    if pneumonia_count > len(pneumonia_patients):
        print(f"  ⚠  Requested {pneumonia_count} pneumonia patients but only "
              f"{len(pneumonia_patients)} exist. Using all of them.")
        pneumonia_count = len(pneumonia_patients)

    if normal_count > len(normal_patients):
        print(f"  ⚠  Requested {normal_count} normal patients but only "
              f"{len(normal_patients)} exist. Using all of them.")
        normal_count = len(normal_patients)

    # ── Step 4: Random stratified sampling ───────────────────────────────────
    selected_pneumonia = random.sample(pneumonia_patients, pneumonia_count)
    selected_normal    = random.sample(normal_patients,    normal_count)

    print(f"\n[3] Selected {len(selected_pneumonia)} Pneumonia + {len(selected_normal)} Normal patients")
    print(f"    Total: {len(selected_pneumonia) + len(selected_normal)} images")

    # ── Step 5: Optional DICOM file existence check ───────────────────────────
    if images_dir and images_dir.exists():
        print("\n[4] Verifying DICOM files exist...")
        missing = []
        all_ids = selected_pneumonia + selected_normal
        for pid in all_ids:
            dcm_path = images_dir / f"{pid}.dcm"
            if not dcm_path.exists():
                missing.append(pid)
        if missing:
            print(f"  ⚠  {len(missing)} DICOM files missing! ({missing[:3]}...)")
        else:
            print(f"  ✓  All {len(all_ids)} DICOM files verified.")
    else:
        print("\n[4] DICOM file verification skipped (images_dir not provided).")

    # ── Step 6: Build full dataframe and split ────────────────────────────────
    pneumonia_df = pd.DataFrame({"patientId": selected_pneumonia, "label": 1})
    normal_df    = pd.DataFrame({"patientId": selected_normal,    "label": 0})
    full_df      = pd.concat([pneumonia_df, normal_df], ignore_index=True)
    full_df      = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

    # Stratified split: separate by class, then split each
    pneu_df = full_df[full_df["label"] == 1].reset_index(drop=True)
    norm_df = full_df[full_df["label"] == 0].reset_index(drop=True)

    pneu_train_size = int(len(pneu_df) * train_ratio)
    norm_train_size = int(len(norm_df) * train_ratio)

    train_df = pd.concat([
        pneu_df.iloc[:pneu_train_size],
        norm_df.iloc[:norm_train_size],
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    val_df = pd.concat([
        pneu_df.iloc[pneu_train_size:],
        norm_df.iloc[norm_train_size:],
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n[5] Split:")
    print(f"    Train: {len(train_df)} images  "
          f"(Pneumonia: {(train_df['label']==1).sum()}, Normal: {(train_df['label']==0).sum()})")
    print(f"    Val:   {len(val_df)} images  "
          f"(Pneumonia: {(val_df['label']==1).sum()},   Normal: {(val_df['label']==0).sum()})")

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = output_dir / "train_subset.csv"
    val_csv_path   = output_dir / "val_subset.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path,     index=False)

    # Save summary
    summary = {
        "created_at":       datetime.now().isoformat(),
        "random_seed":      seed,
        "total_images":     len(full_df),
        "train_images":     len(train_df),
        "val_images":       len(val_df),
        "train_pneumonia":  int((train_df["label"] == 1).sum()),
        "train_normal":     int((train_df["label"] == 0).sum()),
        "val_pneumonia":    int((val_df["label"] == 1).sum()),
        "val_normal":       int((val_df["label"] == 0).sum()),
        "train_ratio":      train_ratio,
        "source_csv":       str(labels_csv),
        "output_dir":       str(output_dir),
    }
    with open(output_dir / "subset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[6] Saved outputs to: {output_dir}/")
    print(f"    ├── train_subset.csv  ({len(train_df)} rows)")
    print(f"    ├── val_subset.csv    ({len(val_df)} rows)")
    print(f"    └── subset_summary.json")
    print("\n✓ Subset preparation complete!")

    return train_df, val_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]   # Two levels up from src/segmentation/
    rsna_root    = project_root / "data" / "rsna_pneumonia"
    images_dir   = rsna_root / "stage_2_train_images"

    train_df, val_df = prepare_balanced_subset(
        labels_csv      = rsna_root / "stage_2_train_labels.csv",
        output_dir      = rsna_root / "subset",
        pneumonia_count = PNEUMONIA_COUNT,
        normal_count    = NORMAL_COUNT,
        train_ratio     = TRAIN_RATIO,
        seed            = RANDOM_SEED,
        images_dir      = images_dir if images_dir.exists() else None,
    )

    print("\nFirst 5 training rows:")
    print(train_df.head())
    print("\nFirst 5 validation rows:")
    print(val_df.head())
