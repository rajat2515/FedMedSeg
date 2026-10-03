<div align="center">

# 🫁 FedMedSeg

### Privacy-Preserving Federated Medical Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-flwr-green?style=for-the-badge)](https://flower.dev)
[![Opacus](https://img.shields.io/badge/Opacus-DP--SGD-blueviolet?style=for-the-badge)](https://opacus.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**FedMedSeg** is a research framework that trains a **MobileNetV2-UNet** segmentation model across multiple simulated hospital clients using **Federated Learning**, with optional **Differential Privacy (DP-SGD)** — so patient X-rays never leave the hospital.

[**Demo**](#-streamlit-inference-portal) · [**Architecture**](#-architecture) · [**Experiments**](#-federated-learning-experiments) · [**Results**](#-results) · [**Quick Start**](#-quick-start)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Federated Learning Experiments](#-federated-learning-experiments)
- [Differential Privacy](#-differential-privacy)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Running Experiments](#-running-experiments)
- [Streamlit Inference Portal](#-streamlit-inference-portal)
- [Results](#-results)
- [Key Concepts](#-key-concepts)
- [References](#-references)

---

## 🔍 Overview

FedMedSeg simulates a real-world federated medical AI scenario: two hospitals collaborate to train a pneumonia segmentation model **without sharing any patient data**. Only model weights are transmitted across the network.

The project covers a complete 5-phase research pipeline:

| Phase | Description |
|-------|-------------|
| **1** | Centralized baseline — MobileNetV2 classifier (upper bound) |
| **2** | Semantic segmentation — MobileNetV2-UNet on RSNA Pneumonia dataset |
| **3** | Federated Learning — FedAvg & FedProx with Non-IID hospital splits |
| **4** | Differential Privacy — DP-FedProx with formal `(ε, δ)` guarantees |
| **5** | Streamlit inference portal — interactive demo for doctors & examiners |

### Key Features

- 🏥 **Non-IID data simulation** — Client A (specialist hospital, 75% pneumonia) vs Client B (general clinic, 75% normal)
- 🔒 **Privacy by design** — Patient X-rays never leave the local client
- 🛡️ **Formal DP guarantees** — DP-SGD via Opacus with tracked ε budget
- ⚡ **No Ray required** — Pure `multiprocessing`-based federation, works on a single machine
- 📊 **Comprehensive evaluation** — Dice, IoU, and Pixel Accuracy across all experiments
- 🖥️ **Interactive demo** — Streamlit app for real-time segmentation inference

---

## 🏗️ Architecture

### MobileNetV2-UNet

The segmentation backbone is a **U-Net** with a **MobileNetV2** encoder pre-trained on ImageNet and fine-tuned on chest X-rays.

```
Input: (B, 3, 224, 224)
│
├── Encoder: MobileNetV2 (features tapped at blocks 1, 3, 6, 13, 17)
│   ├── Block 1  → 112×112 ×16  ch  — low-level edges & textures
│   ├── Block 3  →  56×56  ×24  ch  — simple patterns
│   ├── Block 6  →  28×28  ×32  ch  — medium-level structures
│   ├── Block 13 →  14×14  ×96  ch  — high-level organ shapes
│   └── Block 17 →   7×7   ×320 ch  — bottleneck / deepest features
│
├── Decoder: 4× DecoderBlock (ConvTranspose2d + Skip + Conv-BN-ReLU ×2)
│   ├──  7×7  → 14×14
│   ├── 14×14 → 28×28
│   ├── 28×28 → 56×56
│   └── 56×56 → 112×112 → 224×224
│
└── Output Head: Conv2d(16→1) + Sigmoid
    → (B, 1, 224, 224)  — per-pixel pneumonia probability map
```

**Skip connections** copy encoder feature maps directly into the decoder, preserving fine spatial details (exact edges of the pneumonia opacity) that would otherwise be lost during downsampling.

### Loss Function — Dice-BCE Hybrid

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i \cdot g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

- **BCE** provides pixel-level precision and strong gradients.
- **Dice** penalizes failure to overlap the pneumonia region, preventing the model from predicting "all healthy" on the class-imbalanced dataset.

---

## 📦 Dataset

**RSNA Pneumonia Detection Challenge** ([Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge))

| Property | Value |
|----------|-------|
| Format | DICOM (`.dcm`) |
| Annotations | Bounding boxes → converted to binary masks |
| Total subset | 5,000 images (2,500 Pneumonia + 2,500 Normal) |
| Split | 80% Train (4,000) / 20% Val (1,000) |
| Resolution | 224 × 224 (resized from ~1024 × 1024) |

### Bounding Box → Binary Mask Conversion

Since RSNA provides bounding boxes (not pixel-perfect masks), each box is rendered as a filled white rectangle on a 224 × 224 black canvas. Normal cases receive an all-black (empty) mask.

### Non-IID Client Partitioning

| Client | Role | Pneumonia | Normal | Total |
|--------|------|:---------:|:------:|:-----:|
| **Client A** | Specialist Hospital | 1,500 (75%) | 500 (25%) | 2,000 |
| **Client B** | General Clinic | 500 (25%) | 1,500 (75%) | 2,000 |

This label skew is the key challenge that FedProx is designed to solve.

---

## 🔬 Federated Learning Experiments

Three experiments tell a complete scientific story:

### Experiment 1 — Isolated Training

Each hospital trains **independently** on its own biased data. Client A over-fits to pneumonia; Client B over-fits to normal. Neither generalises.

```bash
python run_isolated.py --device auto --epochs 20
```

### Experiment 2 — FedAvg (McMahan et al., 2017)

Standard Federated Averaging. Hospitals share weights (not data) and the server performs a weighted average each round:

$$w_{r+1} = \sum_{k=1}^{K} \frac{n_k}{n_{\text{total}}} \cdot w_k^r$$

```bash
python run_fedavg.py --rounds 20 --local-epochs 1 --device auto
```

### Experiment 3 — FedProx (Li et al., 2020)

Adds a **proximal regularisation term** to the client loss to prevent client drift on Non-IID data:

$$\mathcal{L}_{\text{prox}}(w_k) = \mathcal{L}_{\text{task}}(w_k) + \frac{\mu}{2}\|w_k - w^r\|^2$$

The μ penalty keeps local updates anchored to the global model, reducing divergence between clients with very different data distributions.

```bash
python run_fedprox.py --rounds 20 --local-epochs 1 --mu 0.01 --device auto
```

### Communication Architecture

```
Server (0.0.0.0:8080)
    │
    ├── Send global weights ──► Client A (127.0.0.1:8080)
    │                               │  Local train (1 epoch, Non-IID)
    │◄── Return updated weights ────┤
    │
    ├── Send global weights ──► Client B (127.0.0.1:8080)
    │                               │  Local train (1 epoch, Non-IID)
    │◄── Return updated weights ────┤
    │
    └── Aggregate → w_{r+1}  →  Evaluate on shared val set
```

Clients run as separate `multiprocessing.Process` workers. A shared `mp.Lock` serialises GPU access to prevent memory deadlocks on a single machine.

---

## 🛡️ Differential Privacy

### The Privacy Threat

An adversary who intercepts model weights can perform **Model Inversion Attacks** to partially reconstruct training images. DP-SGD provides a mathematical guarantee against this.

### DP-SGD — Two-Step Protection (Opacus)

**Step 1 — Gradient Clipping** (bound sensitivity):

$$\bar{g}_i = g_i \cdot \min\!\left(1,\, \frac{C}{\|g_i\|_2}\right)$$

**Step 2 — Gaussian Noise Addition** (masking):

$$\tilde{g} = \frac{1}{B}\left(\sum_{i=1}^{B} \bar{g}_i + \mathcal{N}(0,\, \sigma^2 C^2 \mathbf{I})\right)$$

After training, the privacy budget is reported as:
> *"Patient data is (ε, δ)-differentially private"* — an attacker cannot determine with certainty whether any individual's X-ray was used in training.

```bash
python run_dp_fedprox.py --rounds 20 --epsilon 8.0 --max-grad-norm 1.0 --mu 0.01
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epsilon` | `8.0` | Privacy budget ε (lower = stronger privacy) |
| `--max-grad-norm` | `1.0` | Gradient clipping norm C |
| `--mu` | `0.01` | FedProx proximal coefficient |

---

## 📁 Project Structure

```
FedMedSeg/
├── src/
│   └── segmentation/
│       ├── model_unet.py           # MobileNetV2-UNet architecture
│       ├── dataset_rsna.py         # RSNA DICOM data loader
│       ├── loss.py                 # DiceLoss & DiceBCELoss
│       ├── metrics.py              # Dice, IoU, Pixel Accuracy
│       ├── train_segmentation.py   # Centralised training loop
│       ├── partition_data.py       # Non-IID data splitter
│       ├── fl_client.py            # Flower NumPyClient wrapper
│       ├── fl_server.py            # FedAvg / FedProx strategies
│       ├── privacy.py              # Opacus DP-SGD helpers
│       ├── quantization.py         # INT8 model quantisation
│       ├── prepare_subset.py       # 5k balanced subset extractor
│       └── device_utils.py         # CUDA/CPU/MPS auto-detection
│
├── models/
│   ├── model_1_baseline.py         # Phase 1 CNN classifier
│   ├── model2.py                   # Phase 2 intermediate model
│   └── model3.py                   # Phase 2 MobileNetV2-UNet
│
├── run_isolated.py                 # Experiment 1 — Isolated training
├── run_fedavg.py                   # Experiment 2 — FedAvg
├── run_fedprox.py                  # Experiment 3 — FedProx
├── run_dp_fedprox.py               # Experiment 4 — DP-FedProx
├── run_federation_comparison.py    # Cross-experiment comparison plots
├── run_ablation_comparison.py      # Ablation study runner
├── run_final_comparision.py        # Final result aggregation
│
├── app.py                          # Streamlit inference portal (Phase 5)
├── start_server.py                 # Flower server launcher
├── start_client.py                 # Flower client launcher
├── continuous_server.py            # Persistent server (multi-machine FL)
├── continuous_client.py            # Persistent client (multi-machine FL)
│
├── data/
│   └── rsna_pneumonia/
│       └── subset/                 # client_a_train.csv, client_b_train.csv, val_subset.csv
│
├── checkpoints/                    # Saved model weights (.pth)
├── results/                        # Per-experiment metrics, plots, JSON reports
├── notebooks/                      # Jupyter exploration notebooks
├── requirements.txt
└── track.md                        # Detailed technical implementation notes
```

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/FedMedSeg.git
cd FedMedSeg

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the RSNA Dataset

1. Accept the competition terms on [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
2. Download and extract into `data/rsna_pneumonia/`:

```
data/rsna_pneumonia/
├── stage_2_train_images/     ← DICOM files
└── stage_2_train_labels.csv
```

### 3. Prepare the 5k Balanced Subset & Client Splits

```bash
python -c "
from src.segmentation.prepare_subset import prepare_balanced_subset
prepare_balanced_subset()

from src.segmentation.partition_data import partition_non_iid
partition_non_iid()
"
```

This generates the three CSVs in `data/rsna_pneumonia/subset/`.

---

## 🚀 Running Experiments

### Centralised Baseline (Phase 2)

```bash
python src/segmentation/train_segmentation.py
```

### Federated Experiments (Phase 3)

```bash
# Standard FedAvg
python run_fedavg.py --rounds 20 --local-epochs 1

# FedProx (recommended — better for Non-IID)
python run_fedprox.py --rounds 20 --local-epochs 1 --mu 0.01

# Isolated (shows what happens without federation)
python run_isolated.py
```

### Differentially Private Federated Training (Phase 4)

```bash
python run_dp_fedprox.py --rounds 20 --epsilon 8.0 --max-grad-norm 1.0 --mu 0.01
```

### Generate Comparison Plots

```bash
python run_federation_comparison.py
```

Results (CSV, JSON reports, and PDF plots) are saved under `results/`.

### Optional: Warm-start from a Pre-trained Checkpoint

```bash
python run_fedavg.py --warm-start --rounds 20
```

---

## 🖥️ Streamlit Inference Portal

An interactive web app allows doctors or examiners to upload a chest X-ray and receive real-time segmentation predictions.

```bash
.venv/bin/streamlit run app.py
# Open: http://localhost:8501
```

**Capabilities:**
- Upload `.jpg`, `.png`, or `.dcm` X-ray files
- Side-by-side display: original image, predicted mask, and colour overlay
- Switch between the **Centralized** and **FedProx** model for comparison
- Browse full experiment metrics and convergence curves

---

## 📊 Results

### Segmentation Performance Comparison

| Model | Val Dice ↑ | Val IoU ↑ | Val PixAcc ↑ | Privacy |
|-------|:----------:|:---------:|:------------:|:-------:|
| Isolated (Client A only) | ~0.42 | ~0.28 | ~0.81 | ✅ Local only |
| Isolated (Client B only) | ~0.38 | ~0.24 | ~0.78 | ✅ Local only |
| **FedAvg** | ~0.61 | ~0.45 | ~0.88 | ✅ No data sharing |
| **FedProx (μ=0.01)** | ~0.67 | ~0.51 | ~0.90 | ✅ No data sharing |
| **DP-FedProx (ε=8.0)** | ~0.64 | ~0.48 | ~0.89 | ✅ Formal DP guarantee |
| Centralised (upper bound) | ~0.71 | ~0.56 | ~0.92 | ❌ Data pooled |

> **Key Finding:** FedProx recovers ~94% of centralised performance while keeping patient data at the source. DP-FedProx adds a formal mathematical privacy guarantee with only a ~4% accuracy cost.

---

## 📖 Key Concepts

| Term | Definition |
|------|-----------|
| **Federated Learning (FL)** | Training across distributed institutions without centralising data |
| **FedAvg** | Federated Averaging — baseline FL algorithm (McMahan et al., 2017) |
| **FedProx** | Extension of FedAvg with proximal regularisation for Non-IID data (Li et al., 2020) |
| **Non-IID** | Non-Independent & Identically Distributed — each client has a different data distribution |
| **Client Drift** | Divergence of local models due to heterogeneous data |
| **DP-SGD** | Differentially Private SGD — clips gradients & adds noise for privacy |
| **Privacy Budget (ε)** | Lower ε = stronger privacy guarantee |
| **Dice Coefficient** | Primary segmentation metric — measures mask overlap (0 → 1) |
| **IoU / Jaccard** | Stricter overlap metric: `TP / (TP + FP + FN)` |
| **Skip Connection** | Shortcut paths in U-Net that preserve fine spatial details |
| **DICOM** | Standard medical image format (`.dcm`) |

---

## 📚 References

1. **McMahan, H. B., et al.** (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS. → **FedAvg**

2. **Li, T., et al.** (2020). *Federated Optimization in Heterogeneous Networks*. MLSys. → **FedProx**

3. **Abadi, M., et al.** (2016). *Deep Learning with Differential Privacy*. CCS. → **DP-SGD**

4. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. → **U-Net**

5. **Sandler, M., et al.** (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR. → **MobileNetV2**

6. **Flower Framework** — [flower.dev](https://flower.dev) — Open-source Federated Learning library.

7. **RSNA Pneumonia Detection Challenge** — [Kaggle Dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

---

<div align="center">

Made with ❤️ for privacy-preserving medical AI research

</div>
