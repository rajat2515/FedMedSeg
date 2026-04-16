# src/segmentation/dataset_rsna.py
# FedMedSeg Phase 2 — RSNA Pneumonia Dataset Loader
#
# DATASET FORMAT (RSNA Pneumonia Detection Challenge):
#   data/rsna_pneumonia/
#   ├── stage_2_train_images/          # DICOM files (.dcm) — one per patient
#   │   ├── <patientId>.dcm
#   │   └── ...
#   └── stage_2_train_labels.csv       # Bounding box annotations
#       Columns: patientId | x | y | width | height | Target
#       - Target = 1: Pneumonia (has bounding box)
#       - Target = 0: Normal    (no bounding box, mask is all zeros)
#       Note: One patient can have MULTIPLE bounding boxes (rows) if
#             pneumonia appears in multiple lung regions.
#
# MASK GENERATION LOGIC:
#   1. Create black image 224×224 (all zeros)
#   2. For each bounding box of a patient:
#      a. Original DICOM images are 1024×1024
#      b. Scale coordinates: x_scaled = x * (224/1024)
#      c. Draw filled white rectangle at scaled coordinates
#   3. Normal patients → all-black mask (no infection)
#
# AUGMENTATION — CRITICAL:
#   Image  augmentations: flip, rotate, brightness, normalize
#   Mask   augmentations: flip, rotate ONLY  (same transform as image)
#   Mask must NEVER be normalized or color-jittered — it is binary 0/1.

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


# DICOM is the standard medical image format — requires pydicom to read.
# Install with: pip install pydicom
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
DICOM_ORIGINAL_SIZE = 1024   # All RSNA DICOM images are 1024×1024
TARGET_SIZE         = 224    # We resize to 224×224 for MobileNetV2
SCALE_FACTOR        = TARGET_SIZE / DICOM_ORIGINAL_SIZE   # 224/1024 = 0.21875


def load_dicom_as_pil(dcm_path: Path) -> Image.Image:
    """
    Load a DICOM file (.dcm) and return a PIL RGB Image.

    DICOM images are grayscale (1 channel).
    We convert to RGB because MobileNetV2 expects 3-channel input.

    Args:
        dcm_path (Path): Path to the .dcm file.

    Returns:
        PIL.Image: RGB image.

    Raises:
        ImportError: If pydicom is not installed.
        FileNotFoundError: If the DICOM file doesn't exist.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required to load DICOM files.\n"
            "Install with:  pip install pydicom"
        )

    dcm  = pydicom.dcmread(str(dcm_path))
    arr  = dcm.pixel_array.astype(np.uint8)   # Convert to 8-bit grayscale
    img  = Image.fromarray(arr).convert("RGB") # Grayscale → RGB (3 channels)
    return img


def generate_mask(
    rows: pd.DataFrame,
    img_width: int = DICOM_ORIGINAL_SIZE,
    img_height: int = DICOM_ORIGINAL_SIZE,
    target_size: int = TARGET_SIZE,
) -> Image.Image:
    """
    Convert RSNA bounding box annotations into a binary segmentation mask.

    Algorithm:
      1. Start with a black image (all zeros) of size target_size × target_size
      2. For each bounding box row:
         - Scale (x, y, w, h) from original DICOM size to target_size
         - Draw a filled white rectangle on the mask
      3. Return the binary mask image

    Args:
        rows       (DataFrame): Rows of stage_2_train_labels.csv for one patient.
                                May be empty for Normal patients.
        img_width  (int): Original DICOM image width (usually 1024).
        img_height (int): Original DICOM image height (usually 1024).
        target_size (int): Output mask size (224).

    Returns:
        PIL.Image: Binary mask image ('L' mode) of size target_size × target_size.
                   Pixel value 255 = Pneumonia, 0 = Healthy.
    """
    mask = Image.new("L", (target_size, target_size), 0)  # Black canvas

    if rows is None or len(rows) == 0:
        return mask   # All-black mask for Normal patients

    draw = ImageDraw.Draw(mask)
    scale_x = target_size / img_width
    scale_y = target_size / img_height

    for _, row in rows.iterrows():
        if row["Target"] == 0:
            continue  # Normal row — no bounding box needed

        # Original coordinates from the CSV
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]

        # Scale to target_size
        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        x1 = int((x + w) * scale_x)
        y1 = int((y + h) * scale_y)

        # Clamp to image bounds
        x0 = max(0, min(x0, target_size - 1))
        y0 = max(0, min(y0, target_size - 1))
        x1 = max(0, min(x1, target_size))
        y1 = max(0, min(y1, target_size))

        draw.rectangle([x0, y0, x1, y1], fill=255)  # White rectangle

    return mask


# ── Synchronized Augmentation Helpers ────────────────────────────────────────

def synchronized_augment(
    image: Image.Image,
    mask: Image.Image,
    do_hflip: bool = True,
    do_rotate: bool = True,
    max_angle: float = 10.0,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the SAME spatial augmentations to both image and mask.

    Rules:
      - Horizontal flip: 50% probability — applied identically to both.
      - Random rotation: ±max_angle degrees — same angle for both.
      - Color/brightness: applied to IMAGE ONLY (mask must stay binary).

    Args:
        image      (PIL.Image): The chest X-ray.
        mask       (PIL.Image): The binary segmentation mask.
        do_hflip   (bool): Enable random horizontal flip.
        do_rotate  (bool): Enable random rotation.
        max_angle  (float): Maximum rotation angle in degrees.

    Returns:
        Tuple[PIL.Image, PIL.Image]: Augmented (image, mask).
    """
    # Horizontal flip
    if do_hflip and random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    # Rotation — SAME angle for both
    if do_rotate:
        angle = random.uniform(-max_angle, max_angle)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)

    return image, mask


# ── Main Dataset Class ────────────────────────────────────────────────────────

class RSNAPneumoniaDataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge.

    Reads DICOM images, generates binary segmentation masks from bounding boxes,
    and applies synchronized image+mask augmentations.

    Directory structure expected:
        rsna_root/
        ├── stage_2_train_images/  ← DICOM files (.dcm)
        └── stage_2_train_labels.csv

    OR provide a subset CSV (produced by prepare_subset.py):
        subset_csv: path to CSV with columns [patientId, label]

    Args:
        rsna_root   (str|Path): Root directory of the RSNA dataset.
        subset_csv  (str|Path): Path to subset CSV (patientId, label).
        img_transform (callable): Transform applied to the image tensor.
        augment    (bool): Whether to apply training augmentations. Default False.
    """

    def __init__(
        self,
        rsna_root: str,
        subset_csv: str,
        img_transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        self.rsna_root    = Path(rsna_root)
        self.images_dir   = self.rsna_root / "stage_2_train_images"
        self.labels_csv   = self.rsna_root / "stage_2_train_labels.csv"
        self.img_transform = img_transform
        self.augment       = augment

        # Load the full bounding-box labels
        if not self.labels_csv.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {self.labels_csv}\n"
                "Download the RSNA dataset from Kaggle and place it in data/rsna_pneumonia/"
            )

        all_labels = pd.read_csv(self.labels_csv)

        # Build a lookup: patientId → list of bounding box rows
        self.bbox_lookup = {}
        for pid, group in all_labels.groupby("patientId"):
            self.bbox_lookup[pid] = group

        # Load the subset patient IDs
        self.subset_df = pd.read_csv(subset_csv)
        self.patient_ids = self.subset_df["patientId"].tolist()

        print(f"[RSNAPneumoniaDataset] Loaded {len(self.patient_ids)} patients")
        print(f"  Pneumonia: {(self.subset_df['label'] == 1).sum()}")
        print(f"  Normal:    {(self.subset_df['label'] == 0).sum()}")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_tensor (Tensor): Float tensor (3, 224, 224) in range [0, 1]
                                   after ImageNet normalization.
            mask_tensor  (Tensor): Float tensor (1, 224, 224) with values {0, 1}.
        """
        patient_id = self.patient_ids[idx]
        dcm_path   = self.images_dir / f"{patient_id}.dcm"

        # ── Load image ────────────────────────────────────────────────────────
        image = load_dicom_as_pil(dcm_path)
        image = image.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

        # ── Generate mask ─────────────────────────────────────────────────────
        bbox_rows = self.bbox_lookup.get(patient_id, pd.DataFrame())
        mask = generate_mask(bbox_rows)

        # ── Training augmentations ────────────────────────────────────────────
        if self.augment:
            image, mask = synchronized_augment(image, mask)

        # ── Convert mask to tensor {0, 1} ─────────────────────────────────────
        mask_arr    = np.array(mask, dtype=np.float32) / 255.0   # [0, 255] → [0, 1]
        mask_tensor = torch.tensor(mask_arr).unsqueeze(0)          # (1, 224, 224)

        # ── Apply image transform (resize, ToTensor, Normalize) ───────────────
        if self.img_transform:
            image_tensor = self.img_transform(image)
        else:
            import torchvision.transforms as T
            image_tensor = T.ToTensor()(image)

        return image_tensor, mask_tensor


# ── Quick Sanity Check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    # Test mask generation in isolation (no DICOM needed)
    print("=== Testing mask generation ===")

    # Simulate a patient with two bounding boxes
    dummy_rows = pd.DataFrame({
        "patientId": ["test001", "test001"],
        "x":      [200,  500],
        "y":      [300,  400],
        "width":  [150,  200],
        "height": [120,  180],
        "Target": [1,    1],
    })

    mask = generate_mask(dummy_rows, img_width=1024, img_height=1024, target_size=224)
    mask_arr = np.array(mask)

    print(f"Mask shape:      {mask_arr.shape}")
    print(f"Unique values:   {np.unique(mask_arr)}")  # Should be [0, 255]
    print(f"White pixels:    {(mask_arr == 255).sum()}")
    print(f"Black pixels:     {(mask_arr == 0).sum()}")

    # Normal patient — all-black mask
    normal_mask = generate_mask(pd.DataFrame(), target_size=224)
    assert np.array(normal_mask).sum() == 0, "Normal mask should be all zeros!"
    print("Normal patient mask is all zeros ✓")

    # Visualize if matplotlib is available
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(np.zeros((224, 224), dtype=np.uint8), cmap='gray')
        ax1.set_title("Simulated X-ray (placeholder)")
        ax2.imshow(mask_arr, cmap='gray')
        ax2.set_title("Generated Binary Mask\n(white = pneumonia bounding box)")
        plt.tight_layout()
        plt.savefig("mask_generation_test.png", dpi=100)
        print("\nTest visualization saved: mask_generation_test.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("\nMask generation tests passed ✓")
