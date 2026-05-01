# 📋 FedMedSeg Phase 2–4 — Technical Implementation Tracker

> **Purpose:** This document records **every** method, algorithm, formula, and design decision used in the Phase 2 Semantic Segmentation implementation. It is written so that even if you have never seen the code, you can understand *what* was built, *why*, and *how*.

---

## 1. Architecture: MobileNetV2-UNet

### What is U-Net?
U-Net is a neural network shaped like the letter **"U"**. It has two halves:
- **Encoder (Left side / Contracting Path):** Shrinks the image step by step, extracting *what* is in the image (features like edges, textures, shapes).
- **Decoder (Right side / Expanding Path):** Expands those features back to the original image size, deciding *where* each pixel belongs (infected or healthy).

### Why MobileNetV2 as the Encoder?
Instead of building the Encoder from scratch, we reuse the **MobileNetV2** model (the "champion" from Phase 1). MobileNetV2 already knows how to extract meaningful features from chest X-rays because it was pre-trained on ImageNet and fine-tuned on our pneumonia dataset.

### The Skip Connections (The Key Innovation)
When the Encoder shrinks the image from `224×224` → `112` → `56` → `28` → `14` → `7`, it loses fine spatial details (exact edges of the pneumonia). **Skip connections** copy the feature maps from the Encoder and paste them into the Decoder at the same resolution level. This gives the Decoder both:
- **High-level understanding** (from the deep layers): "This region is pneumonia"
- **Fine spatial details** (from the early layers): "The edge of the pneumonia is exactly here"

### MobileNetV2 Feature Extraction Points
We tap into MobileNetV2's `features` module at specific block indices to get maps at different resolutions:

| Skip | MobileNetV2 Block Index | Output Resolution | Channels | What It Captures |
|------|------------------------|-------------------|----------|------------------|
| Skip 1 | Block 1 (index 1) | 112 × 112 | 16 | Low-level edges, textures |
| Skip 2 | Block 3 (index 3) | 56 × 56 | 24 | Simple patterns |
| Skip 3 | Block 6 (index 6) | 28 × 28 | 32 | Medium-level structures |
| Skip 4 | Block 13 (index 13) | 14 × 14 | 96 | High-level organ shapes |
| Bottleneck | Block 17 (index 17) | 7 × 7 | 320 | Deepest abstract features |

### Decoder Blocks
Each decoder block performs:
1. **Transposed Convolution (`ConvTranspose2d`)**: Doubles the spatial size (e.g., `7×7` → `14×14`).
2. **Concatenation with Skip**: The skip connection feature map is concatenated channel-wise.
3. **Two Conv-BN-ReLU layers**: Refine the combined features.

```
DecoderBlock(in_ch, skip_ch, out_ch):
    ConvTranspose2d(in_ch, out_ch, kernel=2, stride=2)   # Upsample ×2
    Concatenate(upsampled, skip_features)                  # Add spatial detail
    Conv2d(out_ch + skip_ch, out_ch, kernel=3, padding=1)  # Refine
    BatchNorm2d(out_ch)
    ReLU()
    Conv2d(out_ch, out_ch, kernel=3, padding=1)            # Refine more
    BatchNorm2d(out_ch)
    ReLU()
```

### Final Output
A `1×1 Convolution` maps the last decoder output (16 channels) to 1 channel, then a **Sigmoid** activation squeezes every pixel value to `[0, 1]`:
- Values close to **1.0** = Pneumonia (infected pixel)
- Values close to **0.0** = Normal (healthy pixel)

**Output Shape:** Input `(B, 3, 224, 224)` → Output `(B, 1, 224, 224)` — a probability mask the same size as the input image.

---

## 2. Loss Function: Dice-BCE Hybrid Loss

### The Problem with BCE Alone
Standard **Binary Cross-Entropy (BCE)** treats every pixel independently. In chest X-rays, most of the image is healthy lung tissue (~85-90%), so a lazy model can score well just by predicting "healthy" everywhere. BCE cannot punish this behavior strongly enough.

### Dice Loss — The Solution

**Dice Coefficient** measures the *overlap* between the predicted mask (P) and the ground truth mask (G):

$$
\text{Dice}(P, G) = \frac{2 \times |P \cap G|}{|P| + |G|}
$$

In differentiable (soft) form for training:

$$
\text{Dice}(P, G) = \frac{2 \sum_{i} p_i \cdot g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}
$$

Where:
- $p_i$ = predicted probability for pixel $i$ (between 0 and 1)
- $g_i$ = ground truth label for pixel $i$ (0 or 1)
- $\epsilon$ = small smoothing constant (1e-6) to prevent division by zero

**Dice Loss** is then:

$$
\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}(P, G)
$$

- If prediction perfectly overlaps truth → Dice = 1.0 → Loss = 0
- If prediction is completely wrong → Dice ≈ 0 → Loss ≈ 1

### The Hybrid: Dice + BCE

$$
\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}
$$

- **BCE** handles individual pixel accuracy (sharp gradients for easy learning).
- **Dice** handles regional overlap (prevents the model from ignoring small pneumonia regions).

Together, they balance pixel-level precision with region-level accuracy.

---

## 3. Evaluation Metrics

### 3.1 Dice Coefficient (F1 for Segmentation)

$$
\text{Dice} = \frac{2 \times TP}{2 \times TP + FP + FN}
$$

Where (at the pixel level):
- **TP** (True Positive): Pixel correctly identified as Pneumonia
- **FP** (False Positive): Healthy pixel wrongly marked as Pneumonia
- **FN** (False Negative): Pneumonia pixel missed by the model

**Score Range:** 0 (no overlap) to 1 (perfect overlap).
**Target for this project:** Dice > 0.65 is good, > 0.75 is excellent.

### 3.2 Intersection over Union (IoU / Jaccard Index)

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}
$$

IoU is stricter than Dice — it penalizes errors more harshly. The relationship is:

$$
\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}
$$

**Score Range:** 0 to 1. Always lower than Dice for the same prediction.

### 3.3 Pixel Accuracy

$$
\text{PixelAccuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Pixels}}{\text{Total Pixels}}
$$

Simple but can be misleading (a model predicting all pixels as "healthy" could still get ~90% accuracy if only 10% of pixels are infected). That is why we use Dice/IoU as the **primary** metrics and Pixel Accuracy as a **secondary** sanity check.

---

## 4. Dataset: RSNA Pneumonia Detection Challenge

### Why Not the Kermany Dataset?
The Kermany dataset (used in Phase 1) only has **image-level labels** ("Pneumonia" or "Normal"). It does NOT tell us *where* in the image the pneumonia is. For segmentation, we need **pixel-level labels** (masks).

### RSNA Dataset Structure
- **Images:** DICOM format (`.dcm`) — the standard medical imaging format.
- **Labels:** A CSV file (`stage_2_train_labels.csv`) with columns:
  - `patientId` — links to the DICOM filename
  - `x, y, width, height` — bounding box around the pneumonia opacity
  - `Target` — 1 (Pneumonia) or 0 (Normal)

### Bounding Box → Binary Mask Conversion
Since the RSNA dataset provides bounding boxes (not pixel-perfect masks), we convert them:

```
1. Create a black image (all zeros) of size 224×224
2. For each bounding box in the patient's record:
   a. Scale the coordinates from original DICOM size (usually 1024×1024) to 224×224
   b. Draw a filled white rectangle at the scaled coordinates
3. The result is a binary mask where:
   - White (1) = Pneumonia region
   - Black (0) = Healthy region
```

For **Normal** patients (Target=0), the mask is entirely black (no infection).

> **Note:** Bounding boxes are an approximation — a real pneumonia region is irregularly shaped, but boxes give the U-Net enough signal to learn the general location and shape.

### 5,000 Image Subset Strategy
- 2,500 images with `Target=1` (Pneumonia, have bounding boxes)
- 2,500 images with `Target=0` (Normal, all-black masks)
- Split: 80% Training (4,000) / 20% Validation (1,000)

---

## 5. Data Augmentation (Synchronized)

For segmentation, augmentations MUST be applied **identically** to both the image and its mask. If you rotate the X-ray by 15°, you must rotate the mask by exactly 15° too.

| Augmentation | Why | Image | Mask |
|--------------|-----|-------|------|
| Random Horizontal Flip | Lungs are roughly symmetric | Applied | Applied (same flip) |
| Random Rotation (±10°) | X-ray angle varies by machine | Applied | Applied (same angle) |
| Random Brightness/Contrast | Different X-ray machines have different exposures | Applied | **NOT** applied (mask is binary) |
| Resize to 224×224 | Standardize input size | Applied | Applied |
| Normalize (ImageNet stats) | MobileNetV2 expects ImageNet-normalized input | Applied | **NOT** applied |

---

## 6. Training Strategy

### Transfer Learning: Encoder Initialization
The MobileNetV2 Encoder is loaded with **ImageNet pre-trained weights** (same as Phase 1). This means:
- The Encoder already understands edges, textures, and shapes.
- We **freeze** the Encoder for the first few epochs so only the Decoder learns.
- After the Decoder stabilizes, we **unfreeze** the last few Encoder blocks for fine-tuning.

### Optimizer: Adam
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

Adam adapts the learning rate per-parameter using running averages of the gradient ($m$) and squared gradient ($v$). We use:
- Learning Rate: `1e-4` (conservative for medical imaging)
- Weight Decay: `1e-5` (mild regularization)

### Learning Rate Scheduler: ReduceLROnPlateau
If the validation Dice score does not improve for `patience=5` epochs, the learning rate is reduced by a factor of 0.5:

$$
\eta_{\text{new}} = \eta_{\text{old}} \times 0.5
$$

This prevents the model from overshooting the optimum.

### Early Stopping
If validation Dice does not improve for `patience=10` epochs, training is stopped to prevent overfitting.

### Checkpointing
The model weights are saved whenever a **new best validation Dice score** is achieved. This ensures we always keep the best model, even if training continues past the peak.

---

## 7. Prediction & Inference Pipeline

When testing on a **new, unseen X-ray**:

```
1. Load the X-ray image
2. Resize to 224×224, convert to RGB, normalize with ImageNet stats
3. Pass through the trained MobileNetV2-UNet
4. Output: 224×224 probability map (each pixel = probability of pneumonia)
5. Threshold at 0.5: pixels > 0.5 → Pneumonia, pixels ≤ 0.5 → Normal
6. Overlay the binary mask on the original X-ray for visualization
```

---

## 8. Results Structure

All outputs are saved to `results/segmentation/`:

| Output | Format | Purpose |
|--------|--------|---------|
| `training_logs.csv` | CSV | Every epoch: loss, dice, iou, pixel_acc for train & val |
| `loss_curves.pdf` | PDF | Publication-ready train vs val loss curves |
| `dice_iou_curves.pdf` | PDF | Dice & IoU progression over training |
| `training_config.json` | JSON | All hyperparameters, dataset info, timestamps |
| `model_evaluation_report.json` | JSON | Final test metrics with mean ± std |
| `best_model_weights.pth` | PyTorch | Best model checkpoint |
| `prediction_samples/` | PNGs | 20+ side-by-side comparisons |
| `prediction_overlay/` | PNGs | Colored mask overlaid on original X-ray |

---

## 9. File Map

| File | What It Contains | Key Classes/Functions |
|------|-----------------|----------------------|
| `src/segmentation/__init__.py` | Package init | — |
| `src/segmentation/model_unet.py` | U-Net architecture | `MobileNetV2UNet`, `DecoderBlock` |
| `src/segmentation/loss.py` | Loss functions | `DiceLoss`, `DiceBCELoss` |
| `src/segmentation/metrics.py` | Evaluation metrics | `dice_coefficient`, `mean_iou`, `pixel_accuracy` |
| `src/segmentation/dataset_rsna.py` | Data loading | `RSNAPneumoniaDataset` |
| `src/segmentation/prepare_subset.py` | Subset extractor | `prepare_balanced_subset()` |
| `src/segmentation/train_segmentation.py` | Training loop | `train_one_epoch()`, `validate()`, `main()` |
| `notebooks/07-RSNA-Data-Preparation.ipynb` | Data inspection | Visual mask generation demo |
| `notebooks/08-Segmentation-Training.ipynb` | Training showcase | Live training with plots |
| `notebooks/09-Segmentation-Evaluation.ipynb` | Results & analysis | Publication-ready evaluation |

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **Encoder** | The "understanding" half of U-Net — shrinks the image and extracts features |
| **Decoder** | The "drawing" half — expands features back to full resolution to create a mask |
| **Skip Connection** | A shortcut that copies early features to later layers to preserve fine details |
| **DICOM (.dcm)** | Digital Imaging and Communications in Medicine — standard medical image format |
| **Binary Mask** | A black-and-white image where white = infected area, black = healthy area |
| **Dice Coefficient** | Measures how much the predicted mask overlaps with the true mask (0 to 1) |
| **IoU (Jaccard)** | Stricter overlap metric — area of intersection divided by area of union |
| **Transposed Convolution** | "Reverse" convolution that increases spatial resolution (upsampling) |
| **Bounding Box** | A rectangle defined by (x, y, width, height) around a region of interest |
| **Sigmoid** | Function that squeezes any number to the range [0, 1] — used for probabilities |

---

## 11. Non-IID Data Partitioning (Phase 3)

### What is Non-IID?
In real hospitals, each institution sees a **different mix of patients**. A specialist centre sees many sick patients; a small clinic sees mostly healthy ones. This means the data at each hospital is **Non-Independent and Identically Distributed (Non-IID)** — it does NOT look like the overall population.

### Our Simulation Strategy
We split the 4,000 training images into 2 clients with **label skew**:

| Client | Role | Pneumonia | Normal | Total | Skew |
|--------|------|-----------|--------|-------|------|
| **Client A** | Specialist Hospital | 1,500 (75%) | 500 (25%) | 2,000 | Heavy Pneumonia |
| **Client B** | General Clinic | 500 (25%) | 1,500 (75%) | 2,000 | Heavy Normal |

### Why This Matters
- If data were IID (same distribution at both hospitals), federation is trivial.
- Non-IID is the **hard problem** — and solving it is the scientific contribution.
- The validation set stays **global** (shared) for fair comparison.

---

## 12. Federated Averaging — FedAvg (McMahan et al., 2017)

### The Core Idea
Instead of sending patient data to a central server (privacy violation!), each hospital:
1. **Receives** a copy of the global model
2. **Trains** the model on its own local data
3. **Sends back** only the updated model weights (NOT the data)
4. The server **averages** the weights to create an improved global model

### Algorithm (Pseudocode)

```
Initialize global model w₀

For each round r = 1, 2, ..., R:
    Server sends w_r to ALL clients

    For each client k in parallel:
        w_k ← LocalTrain(w_r, local_data_k, E epochs)

    Server aggregates:
        w_{r+1} = Σ (n_k / n_total) × w_k

    where:
        n_k     = training samples on client k
        n_total = total samples across all clients
```

### Aggregation Formula

$$
w_{r+1} = \sum_{k=1}^{K} \frac{n_k}{n_{\text{total}}} \cdot w_k^{r}
$$

This is a **weighted average** — clients with more data have more influence.

### Our Configuration
- **Clients:** 2 (Client A + Client B)
- **Rounds:** 20 communication rounds
- **Local Epochs:** 1 epoch per round per client
- **Framework:** Flower (flwr) — the leading open-source FL framework

### Why Use Low Local Epochs? (The Danger of Client Drift)
In traditional deep learning, we train for many epochs (e.g., 20, 50, 100) to ensure the model fully learns the dataset. In **Federated Learning**, training for many local epochs *before* aggregating is actually **harmful**. 

If Client A (75% pneumonia) trains for 10 epochs locally without talking to the server, its model will drastically overfit to the heavy pneumonia bias. It will effectively "forget" the global knowledge about healthy patients. This is known as **Client Drift**.

By keeping the local epochs very low (e.g., **1 or 2 epochs**), the clients take a small, safe step in the direction of their local data, and then immediately sync with the server. This frequent synchronization keeps all hospitals tethered to the global objective and prevents them from wandering off track.

---

## 13. FedProx — Proximal Federated Learning (Li et al., 2020)

### The Problem with FedAvg on Non-IID Data
When client data is very different (Non-IID), each client's local updates pull the global model in **conflicting directions**. This is called **client drift**:
- Client A (75% pneumonia) pushes the model to detect more infections
- Client B (75% normal) pushes the model to predict healthy
- The averaged model oscillates and converges slowly

### FedProx's Solution: The Proximal Term
FedProx modifies the **client-side loss function** by adding a penalty for drifting too far from the global model:

$$
\mathcal{L}_{\text{prox}}(w_k) = \mathcal{L}_{\text{task}}(w_k) + \frac{\mu}{2} \|w_k - w^{r}\|^2
$$

Where:
- $\mathcal{L}_{\text{task}}$ = standard Dice-BCE loss (same as Phase 2)
- $w_k$ = local model weights being updated
- $w^{r}$ = global model weights received at the start of round $r$
- $\mu$ = proximal coefficient (controls regularization strength)
- $\|w_k - w^{r}\|^2$ = squared L2 distance between local and global weights

### Understanding μ (Mu)
| μ Value | Effect | Analogy |
|---------|--------|---------|
| 0.0 | No regularization (= FedAvg) | Free-for-all meeting |
| 0.01 | Mild — recommended default | Meeting with a gentle moderator |
| 0.1 | Strong — local updates are small | Strict moderator |
| 1.0 | Very strong — barely any local learning | Dictator (ignores local data) |

### Implementation in Code
```python
# Inside client's local training loop:
loss = criterion(preds, masks)           # Standard Dice-BCE loss

# Proximal term: (μ/2) × ||w_local - w_global||²
proximal_term = 0.0
for local_param, global_param in zip(model.parameters(), global_params):
    proximal_term += (local_param - global_param).norm(2) ** 2

loss = loss + (mu / 2.0) * proximal_term  # Combined loss
loss.backward()                           # PyTorch handles the rest
```

---

## 14. Federation Experiment Framework

### The Three Experiments
We run three experiments to tell a complete scientific story:

| # | Experiment | What It Proves |
|---|-----------|----------------|
| 1 | **Isolated Training** | Training alone on biased data FAILS |
| 2 | **FedAvg** | Standard federation RECOVERS performance |
| 3 | **FedProx** | Proximal term IMPROVES Non-IID robustness |

### Flower Framework Integration
We use the **Flower** (flwr) framework for federated simulation:
- `fl.client.NumPyClient` — wraps our PyTorch model for FL communication
- `fl.simulation.start_simulation()` — simulates multi-client training on one machine
- `fl.server.strategy.FedAvg` / `FedProx` — configurable aggregation strategies

### Communication Flow Per Round
```
Server                          Client A              Client B
  │                               │                      │
  ├──── Send global weights ─────►├                      │
  ├──── Send global weights ──────┼─────────────────────►│
  │                               │                      │
  │                     Local train (1 epoch)   Local train (1 epoch)
  │                     on Non-IID data          on Non-IID data
  │                               │                      │
  │◄──── Return updated weights ──┤                      │
  │◄──── Return updated weights ──┼──────────────────────┤
  │                               │                      │
  ├── Aggregate: w_new = Σ(n_k/n) × w_k                 │
  │                               │                      │
  └── Evaluate global model on shared validation set ────┘
```

### File Map (Phase 3)

| File | What It Contains | Key Functions |
|------|-----------------|---------------|
| `src/segmentation/partition_data.py` | Non-IID data splitter | `partition_non_iid()` |
| `src/segmentation/fl_client.py` | Flower client wrapper | `FedMedSegClient`, `train_local()` |
| `src/segmentation/fl_server.py` | Server strategies | `create_fedavg_strategy()`, `create_fedprox_strategy()` |
| `run_isolated.py` | Isolated training experiment | `train_client()`, `main()` |
| `run_fedavg.py` | FedAvg experiment | `create_client_fn()`, `main()` |
| `run_fedprox.py` | FedProx experiment | `create_client_fn()`, `main()` |
| `run_federation_comparison.py` | Final comparison plots | `plot_bar_comparison()`, `plot_convergence()` |

---

## 15. Glossary (Phase 3 Additions)

| Term | Meaning |
|------|---------|
| **Federated Learning (FL)** | Training a shared model across multiple institutions without sharing raw data |
| **FedAvg** | Federated Averaging — the baseline FL algorithm that averages model weights |
| **FedProx** | Extension of FedAvg that adds a proximal penalty to handle Non-IID data |
| **Non-IID** | Non-Independent and Identically Distributed — data that differs between clients |
| **Client Drift** | When local updates diverge too much due to data heterogeneity |
| **Proximal Term** | Regularization penalty: (μ/2) × ||w_local - w_global||² |
| **Communication Round** | One cycle of: distribute weights → local training → collect weights → aggregate |
| **Aggregation** | Combining multiple clients' updated weights into one global model |
| **Flower (flwr)** | Open-source federated learning framework for Python |
| **NumPyClient** | Flower's client interface — bridges PyTorch models with FL communication |
| **Label Skew** | A type of Non-IID where different clients have different class proportions |

---

## 16. Differential Privacy — DP-SGD (Abadi et al., 2016) [Phase 4]

### The Privacy Threat in Federated Learning
Federated Learning prevents *direct* sharing of patient X-rays. However, an attacker who intercepts the **model weights** (gradients) can perform a **Model Inversion Attack** to partially reconstruct the original patient images. Differential Privacy (DP) solves this by mathematically guaranteeing that no individual patient's data can be perfectly reverse-engineered.

### DP-SGD (Differentially Private Stochastic Gradient Descent)
Standard training computes gradients and updates weights. DP-SGD intercepts these gradients and modifies them in two steps:

**Step 1. Gradient Clipping (Bounding Sensitivity)**
We must limit how much *any single patient's X-ray* can affect the model. We clip the $L_2$ norm of each per-sample gradient $g_i$ to a maximum threshold $C$ (`max_grad_norm`):

$$
\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)
$$

**Step 2. Noise Addition (Masking)**
After clipping and averaging the gradients across the batch (size $B$), we add random Gaussian noise scaled by the clipping norm $C$ and a noise multiplier $\sigma$:

$$
\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)
$$

The final weight update becomes: $\theta_{t+1} = \theta_t - \eta \tilde{g}$

### The Privacy Budget: ($\epsilon, \delta$)
- **$\epsilon$ (Epsilon):** The privacy budget. Lower is more private, but less accurate.
  - $\epsilon = 1.0$: Extreme privacy (high noise, large accuracy drop).
  - **$\epsilon = 8.0$: Healthcare industry standard (moderate noise, minimal accuracy drop). We use 8.0.**
- **$\delta$ (Delta):** The probability that the privacy guarantee fails. Set to $10^{-5}$ (must be `< 1/N`).

### Implementation (Opacus)
We use Facebook's **Opacus** library, which hooks into PyTorch's autograd engine to perform the clipping and noise injection dynamically. The model, optimizer, and data loader are wrapped in an `Opacus PrivacyEngine`.

---

## 17. Model Quantization [Phase 4]

### The Efficiency Problem
Our MobileNetV2-UNet has ~3.2 million parameters. As standard 32-bit floats (`float32`), this takes **12.8 MB** per client per round.
- 2 clients × 20 rounds = 512 MB of total bandwidth used.
For resource-constrained clinics with slow internet, this is a bottleneck.

### Post-Training Dynamic Quantization (PTDQ)
We compress the model weights from 32-bit floats down to **8-bit integers (`int8`)** right before transmitting them over the network.

- **Post-Training**: No need to retrain the model.
- **Dynamic**: Ony `Linear` (Dense bottleneck) layers are quantized. Activations remain `float32` during actual computation.
- **Size Savings**: Reduces the communication payload from ~12.8 MB to **~3.2 MB** (a ~75% reduction in bandwidth overhead).

---

## 18. Continuous Learning Pipeline (MLOps) [Phase 5]

### The Static vs. Dynamic Problem
Currently, the FL system is static — we run it once, and it stops. In reality, hospitals continuously acquire new X-rays. 

### The 24-Hour Cycle Architecture
We use a background scheduler (`APScheduler`) to automate retraining without human intervention.

1. **The Scheduler** wakes up daily at 2:00 AM.
2. **The Data Watcher** scans the data directories to count files.
   - If `current_count > previous_count`, a retraining trigger fires.
3. The server starts a new **DP-FedProx** simulation specifically on the newly acquired data, using the best previous model as a _warm start_.
4. **The Model Registry** evaluates the freshly trained model.
   - If the new Dice score > the old Dice score: The model is saved as the new `current_best.pth`.
   - If the new model performs worse (e.g., catastrophic forgetting due to anomalous data): The update is rejected.

---

## 19. Inference Web Portal — Streamlit UI [Phase 5]

### Purpose
A doctor, examiner, or stakeholder can upload any chest X-ray and receive an immediate AI-driven segmentation result. This bridges the gap between the complex backend math and a human-readable, actionable visualization. Without this, the entire system is a "black box" of terminal output.

### Technology: Streamlit
We use **Streamlit** — a Python library that turns a `.py` script into a fully interactive web page in ~50 lines of code. It handles file upload widgets, image rendering, and layout components automatically. No HTML/JavaScript required.

### Application Structure (`app.py`)

The UI has **three tabs**:

#### Tab 1: Run Inference
This is the core functionality.
1. **Upload:** User uploads a `.jpg` / `.png` chest X-ray.
2. **Model loads** from the saved `.pth` checkpoint (`model3c_best.pth`).
3. **Inference Pipeline:**
   - Image is resized to 224×224 and normalized with ImageNet statistics.
   - Passed through the `MobileNetV2UNet` → output is a 224×224 probability map.
   - A Sigmoid activation converts logits to `[0,1]` probabilities.
   - Pixels **above the threshold (default 0.5)** are classified as Pneumonia.
4. **Three visualizations are shown:**
   - **Overlay:** Original X-ray with a red mask drawn over the Pneumonia region.
   - **Heatmap:** Full probability map colored on a blue→red scale (more red = higher probability).
   - **Statistics:** Max probability, mean probability, infected pixel count, and % lung coverage.

#### Tab 2: Experiment Results
Displays the live results from `federation_summary.json` and `dp_fedprox_report.json` as an interactive table. Also renders the saved `federation_comparison.png` and `convergence_curves.png` charts directly in the browser.

#### Tab 3: About
A text summary of all four phases with embedded LaTeX formulas rendered by Streamlit's built-in MathJax support.

### Inference Formula Recap
```
Input X-ray (224×224, RGB)
       ↓
  MobileNetV2-UNet
       ↓
  Logits (224×224, 1 channel)
       ↓
  Sigmoid → Probability Map [0,1]
       ↓
  Threshold at 0.5:  pixel ≥ 0.5 → Pneumonia (1)
                     pixel < 0.5 → Normal (0)
       ↓
  Binary Mask + Overlay + Heatmap
```

### File
| File | Role |
|------|------|
| `app.py` | Main Streamlit application (project root) |

### How to Run
```bash
cd /home/rajat/Documents/Project/FedMedSeg
.venv/bin/streamlit run app.py
# Opens at: http://localhost:8501
```

---

## 20. Real-World Distributed Deployment — Multi-Laptop Testing [Phase 6]

### Motivation
All previous experiments (Phases 3–5) used Python `multiprocessing` or Ray simulation to run the server and both hospital clients on **one single machine**. This is a simulation shortcut — it does NOT test real network communication, OS compatibility, or bandwidth constraints.

With three physical laptops available, we can run a **true distributed federated learning experiment** where every participant is a separate machine communicating over a real local network.

### Hardware Setup

| Role | Machine | OS | Script |
|------|---------|-----|--------|
| **Central Model Initiator (Server)** | Laptop 1 | Linux | `start_server.py` |
| **Hospital Node A (Client A)** | Laptop 2 | Windows | `start_client.py` |
| **Hospital Node B (Client B)** | Laptop 3 | Windows | `start_client.py` |

All three laptops must be on the **same local network** (Wi-Fi or LAN).

---

### Why the New Scripts? (Windows Compatibility)

The old experiment scripts (`run_fedprox.py`, `run_fedavg.py`) used Python `multiprocessing` to spawn child processes on **one machine**. This has two problems:
1. `multiprocessing` with `spawn` context (required on Windows) has different semantics than Linux `fork`.
2. Ray simulation is entirely incompatible with Windows.

The two new entry-point scripts are **single-process network clients/servers** — they use only **Flower's gRPC transport layer** (standard TCP sockets), which is fully cross-platform.

| Script | Uses multiprocessing? | Uses Ray? | Windows-safe? |
|---|---|---|---|
| `run_fedprox.py` (old) | ✅ Yes | ❌ No | ❌ **Not safe** |
| `run_fedavg.py` (old) | ✅ Yes | ❌ No | ❌ **Not safe** |
| `start_server.py` (new) | ❌ No | ❌ No | ✅ **Safe** |
| `start_client.py` (new) | ❌ No | ❌ No | ✅ **Safe** |

---

### Per-Round Communication Flow (Real Network)

```
ROUND N
=========

[Linux Server]              [Windows Client A]       [Windows Client B]
     │                            │                         │
     │──── Send Global Weights ──►│                         │
     │──── Send Global Weights ───────────────────────────►│
     │                            │                         │
     │                    Train on local data       Train on local data
     │                    (client_a dataset)        (client_b dataset)
     │                    + FedProx penalty         + FedProx penalty
     │                            │                         │
     │◄─── Return Updated Weights─│                         │
     │◄─── Return Updated Weights─────────────────────────│
     │                            │                         │
     │   FedAvg Aggregation:                               │
     │   w_global = Σ (n_k / n_total) × w_k               │
     │                            │                         │
     └────────── Repeat Round N+1 ───────────────────────►│
```

**Transport:** All communication is over **Flower gRPC** (HTTP/2 + Protocol Buffers). The model weights (~12.8 MB per round as float32, ~3.2 MB quantized) are serialized as NumPy arrays and transmitted as binary protobuf messages.

---

### Data Privacy Guarantee

Even in the real deployment, **raw patient data (X-ray images) never leaves the hospital laptop**:

```
Linux Server          Windows Client A       Windows Client B
─────────────         ─────────────────      ─────────────────
Global model          Hospital A X-rays      Hospital B X-rays
weights only    ←→    (STAYS LOCAL)    ←→    (STAYS LOCAL)
                      Sends: model           Sends: model
                      weight arrays only     weight arrays only
```

---

### How to Run

**Step 1 — Start the server on Laptop 1 (Linux):**
```bash
# Note your IP first: ip addr show
python start_server.py --host 0.0.0.0 --port 8080 --strategy fedprox --clients 2 --rounds 20
# Server will wait here until both clients connect
```

**Step 2 — Start Hospital Node A on Laptop 2 (Windows):**
```bat
python start_client.py --server <LINUX_IP>:8080 --node-type client_a --mu 0.01
```

**Step 3 — Start Hospital Node B on Laptop 3 (Windows):**
```bat
python start_client.py --server <LINUX_IP>:8080 --node-type client_b --mu 0.01
```

Once both clients connect, the server automatically initiates Round 1 and training proceeds.

**Optional — Run with Differential Privacy:**
```bat
python start_client.py --server <LINUX_IP>:8080 --node-type client_a --mu 0.01 --use-dp --epsilon 8.0
```

---

### File Map (Phase 6)

| File | What It Contains | Key Functions |
|------|-----------------|---------------|
| `start_server.py` | Standalone Flower server entry point | `main()` — configures strategy and starts `fl.server.start_server()` |
| `start_client.py` | Standalone Flower client entry point | `main()` — loads dataset, builds `FedMedSegClient`, calls `fl.client.start_numpy_client()` |

---

## 21. Glossary (Phase 6 Additions)

| Term | Meaning |
|------|---------|
| **gRPC** | Google Remote Procedure Call — the binary network protocol Flower uses to send model weights between server and clients |
| **Non-IID (Real)** | Data heterogeneity that exists *naturally* because different hospitals see different patient populations — no artificial splitting needed in real deployment |
| **Central Model Initiator** | The server laptop that holds the global model, broadcasts weights, and performs FedAvg aggregation |
| **Hospital Node** | A client laptop that holds local patient data, trains locally, and sends weight updates to the server |
| **num_workers=0** | DataLoader setting that disables multiprocessing for data loading — required on Windows to avoid subprocess spawn errors |

---

*Last Updated: 2026-04-29*
*Phase: 6 — Real Multi-Laptop Distributed Deployment*

---

## 22. Continuous Federated Learning Pipeline [Phase 7]

### The Problem: Static Training vs. Real-World Hospitals

The Phase 6 deployment runs a **fixed session** — `N` rounds, then every process exits. In a real hospital network, new X-rays are collected every day. Restarting and manually reconnecting all nodes after every session is not practical at production scale. The **Continuous Pipeline** turns FedMedSeg into a long-running, self-managing training system.

---

### Architecture: Session Loop

Instead of a single training run, the system is organised into **sessions**. A session is one complete batch of `--rounds` federated rounds. After each session the server:
1. Saves a **checkpoint** of the aggregated global model.
2. Logs performance metrics to `pipeline_log.json`.
3. Waits for a configurable **cool-down interval**.
4. Starts the **next session** using the best checkpoint so far as the **warm start**.

```
  Session 1  (R rounds)  →  save checkpoint  →  cool-down
       ↓
  Session 2  (R rounds)  →  save checkpoint  →  cool-down
       ↓
  Session 3  ...             (until Ctrl+C or --sessions N)
```

This means **learning is cumulative** — each session builds on what was learned in all previous sessions, rather than starting from scratch.

---

### Warm-Start (Incremental Learning)

At the beginning of every session the server loads the **best global model** saved by any previous session and uses its weights as `initial_parameters` for the Flower strategy:

$$
w_{\text{init}}^{(s)} = w_{\text{best}}^{(s-1)}
$$

If no checkpoint exists yet (Session 1), the model is initialised with standard ImageNet pre-trained weights (same as before).

This is equivalent to **Continual Learning** in the federated setting — the model never forgets what it learned in earlier sessions even as new data arrives.

---

### Checkpoint Strategy

After each session two files are written/updated:

| File | Updated When | Purpose |
|------|-------------|---------|
| `checkpoints/session_NNN_model.pth` | Every session | Full history — one file per session |
| `checkpoints/best_global_model.pth` | Val Dice improves | Rolling best — used as warm-start next session |

The **best model** criterion is the **global Val Dice** reported by the `weighted_average()` metric aggregation function. If a session's Val Dice does not beat the current best (e.g., due to noisy or anomalous data), `best_global_model.pth` is **not overwritten**, protecting the best known model.

---

### Pipeline Logging — `pipeline_log.json`

A JSON array is maintained at the project root. One entry is appended after every session:

```json
[
  {
    "session": 1,
    "timestamp": "2026-05-01 10:30:00",
    "rounds": 20,
    "total_time_sec": 3420.5,
    "val_dice": 0.7123,
    "val_iou": 0.5538,
    "val_pixel_acc": 0.9201,
    "is_best": true
  },
  ...
]
```

This log can be used to plot convergence across sessions, detect data drift, or audit training history.

---

### Auto-Reconnecting Clients

The original `start_client.py` exits once the server closes the gRPC connection at the end of the final round. In the continuous pipeline the server restarts for Session 2 while the clients are still alive (just idle during the cool-down).

`continuous_client.py` wraps `fl.client.start_numpy_client()` in a **reconnect loop**:

```
while True:
    try:
        fl.client.start_numpy_client(server_address, client)
        # Server closed connection cleanly → session done
        wait(reconnect_delay)       # server is in cool-down
        # next iteration → reconnect for Session N+1
    except ConnectionRefusedError:
        # Server not yet up → retry after reconnect_delay
        wait(reconnect_delay)
```

Clients can be configured with `--reconnect-retries -1` (retry indefinitely) or a fixed budget.

---

### Data-Watch Mode (`--data-watch`)

When `--data-watch` is passed to `continuous_client.py`, the client **rebuilds its `RSNAPneumoniaDataset` and `DataLoader`** at the start of every new session. This means:

- If a hospital administrator appended new patient records to `client_a_train.csv` during the cool-down, they will be included in the next session's training **automatically** — no restart required.
- The previous session's DataLoader is discarded and garbage-collected.

This simulates the real-world scenario where hospitals continuously acquire new diagnostic images.

---

### Checkpoint Helpers Added to `fl_server.py`

Two new functions were added to `src/segmentation/fl_server.py`:

| Function | Purpose |
|----------|---------|
| `save_global_model(parameters, model_shell, save_path)` | Converts Flower `Parameters` → `state_dict` → saves `.pth` |
| `load_global_model_parameters(checkpoint_path, model_shell)` | Loads `.pth` → returns Flower `Parameters` for `initial_parameters` |

These are the bridge between Flower's NumPy-array serialization format and PyTorch's `torch.save` / `torch.load` format.

---

### How to Run the Continuous Pipeline

**On the Linux server laptop:**
```bash
python continuous_server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --rounds 20 \
    --clients 2 \
    --strategy fedprox \
    --mu 0.01 \
    --sessions -1 \          # run indefinitely
    --interval 3600 \        # 1-hour cool-down between sessions
    --checkpoint-dir checkpoints/
```

**On each Windows hospital laptop:**
```bat
python continuous_client.py ^
    --server <LINUX_IP>:8080 ^
    --node-type client_a ^
    --mu 0.01 ^
    --reconnect-retries -1 ^
    --reconnect-delay 30 ^
    --data-watch
```

Press `Ctrl+C` on the server to stop the pipeline gracefully after the current session finishes.

---

### File Map (Phase 7)

| File | What It Contains | Key Functions |
|------|-----------------|---------------|
| `continuous_server.py` | Session-loop server entry point | `main()` — session loop, warm-start, checkpointing, `pipeline_log.json` |
| `continuous_client.py` | Auto-reconnecting hospital node | `main()` — reconnect loop, `build_data_loaders()`, data-watch |
| `src/segmentation/fl_server.py` | *(modified)* Added checkpoint helpers | `save_global_model()`, `load_global_model_parameters()` |
| `checkpoints/best_global_model.pth` | *(auto-generated)* Rolling best global model | — |
| `checkpoints/session_NNN_model.pth` | *(auto-generated)* Per-session checkpoint | — |
| `pipeline_log.json` | *(auto-generated)* JSON log of all sessions | — |

---

## 23. Glossary (Phase 7 Additions)

| Term | Meaning |
|------|---------|
| **Session** | One complete batch of `R` federated rounds. Multiple sessions form the continuous pipeline |
| **Warm Start** | Initialising a new session from the best checkpoint of all previous sessions rather than random weights |
| **Cool-down Interval** | The wait time between sessions (`--interval`). Allows hospitals to upload new data before the next session |
| **Data-Watch** | Client-side mode that rebuilds the DataLoader before each session to include newly arrived patient records |
| **Pipeline Log** | `pipeline_log.json` — a persistent JSON array tracking per-session metrics across all sessions |
| **Incremental Learning** | Learning cumulatively from new data without forgetting knowledge from previous data |
| **Rolling Best** | `best_global_model.pth` — updated only when the current session beats the historical best Val Dice |

---

*Last Updated: 2026-05-01*
*Phase: 7 — Continuous Federated Learning Pipeline*
