# Federated Learning System — Real-World Architecture Guide

## Yes, This Exists — It's Called Federated Learning

What you described is **exactly** Federated Learning (FL), pioneered by Google in 2017 (McMahan et al.). Your vision maps perfectly:

| Your Vision | FL Term |
|-------------|---------|
| Two hospitals on separate laptops | **Clients** (or FL Nodes) |
| Central server | **Aggregation Server** |
| Send model to hospitals | **Model Distribution** (broadcast) |
| Train locally on own data | **Local Training** |
| Send results back | **Weight Upload** (not data — this is the key!) |
| Aggregate and improve | **Federated Averaging (FedAvg)** |
| Repeat continuously | **Communication Rounds** |

---

## How the System Works (Step by Step)

### The Flow — One Communication Round

```
┌─────────────────────────────────────────────────────────────────┐
│                        ROUND 1 OF N                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   STEP 1: Global Server broadcasts model weights                │
│                                                                 │
│            ┌──────────────┐                                     │
│            │ Global Server│                                     │
│            │  (Laptop 3)  │                                     │
│            │  IP: 192.168.│                                     │
│            │     1.100    │                                     │
│            └──┬───────┬───┘                                     │
│         Wi-Fi │       │ Wi-Fi                                   │
│               │       │                                         │
│               ▼       ▼                                         │
│   ┌───────────────┐  ┌───────────────┐                          │
│   │ Hospital A    │  │ Hospital B    │                          │
│   │ (Laptop 1)    │  │ (Laptop 2)    │                          │
│   │ IP: 192.168.  │  │ IP: 192.168.  │                          │
│   │    1.101      │  │    1.102      │                          │
│   └───────────────┘  └───────────────┘                          │
│                                                                 │
│   STEP 2: Each hospital trains on LOCAL data (data never moves) │
│                                                                 │
│   ┌───────────────┐  ┌───────────────┐                          │
│   │ Hospital A    │  │ Hospital B    │                          │
│   │ trains on its │  │ trains on its │                          │
│   │ 1500 X-rays   │  │ 2000 X-rays   │                          │
│   │ (mostly sick) │  │ (mostly healthy│                          │
│   └──────┬────────┘  └──────┬────────┘                          │
│          │                  │                                    │
│   STEP 3: Hospitals send ONLY updated weights back              │
│          │    (NOT patient data!)                                │
│          │                  │                                    │
│          ▼                  ▼                                    │
│            ┌──────────────┐                                     │
│            │ Global Server│                                     │
│            │  Aggregates: │                                     │
│            │  w_new =     │                                     │
│            │  avg(wA, wB) │                                     │
│            └──────────────┘                                     │
│                                                                 │
│   STEP 4: Repeat from Step 1 with improved model                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 3 Machines — What Runs Where

### Machine 1: Global Server (Your Laptop / Desktop)

**Role:** The "brain" — coordinates everything, never sees patient data.

**What it does:**
1. Holds the **global model** (e.g., MobileNetV2-UNet)
2. Sends model weights to all hospitals via network
3. Receives updated weights from hospitals after local training
4. **Aggregates** the updates (weighted average by dataset size)
5. Evaluates the new global model on a held-out test set
6. Repeats for N rounds

**What it needs:**
- Python + Flower server (`flwr`)
- The model architecture definition (so it knows the weight structure)
- A test/validation dataset for evaluation
- Network connectivity to both hospitals

### Machine 2: Hospital A (Laptop 1)

**Role:** A participant — trains on its own patient data privately.

**What it does:**
1. Receives global model weights from the server
2. Loads weights into its local model copy
3. Trains the model on **its own X-ray database** (e.g., 2,000 images)
4. Sends back **only the updated model weights** (a ~18 MB file for your MobileNetV2-UNet)
5. Patient images NEVER leave this machine

**What it needs:**
- Python + Flower client (`flwr`)
- Its own local dataset (X-rays stored locally)
- The model architecture definition
- Network connectivity to the server

### Machine 3: Hospital B (Laptop 2)

**Role:** Same as Hospital A, but with different data.

---

## Network Architecture

### What Protocol Is Used?

Flower uses **gRPC** (Google Remote Procedure Call) under the hood — a high-performance, binary communication protocol commonly used in production systems.

```
                    Wi-Fi Network: 192.168.1.x
                    ─────────────────────────

    ┌────────────┐       gRPC (port 8080)       ┌────────────┐
    │ Hospital A │ ◄──────────────────────────►  │   Global   │
    │ 192.168.   │                               │   Server   │
    │ 1.101      │       Model weights           │ 192.168.   │
    └────────────┘       (~18 MB per round)      │ 1.100      │
                                                 │ Port: 8080 │
    ┌────────────┐       gRPC (port 8080)       │            │
    │ Hospital B │ ◄──────────────────────────►  │            │
    │ 192.168.   │                               └────────────┘
    │ 1.102      │       Model weights
    └────────────┘       (~18 MB per round)
```

### What Gets Sent Over the Network?

| What | Size | Direction |
|------|------|-----------|
| Model weights (NumPy arrays) | ~18 MB | Server → Client |
| Updated weights | ~18 MB | Client → Server |
| Configuration (epochs, LR) | ~1 KB | Server → Client |
| Metrics (loss, dice, iou) | ~100 bytes | Client → Server |
| **Patient Data** | **NEVER** | **NEVER TRANSMITTED** |

> [!IMPORTANT]
> **The entire point**: Patient images, records, and private medical data NEVER leave the hospital laptop. Only mathematical numbers (model weights) are transmitted. This is the privacy guarantee of federated learning.

---

## How To Set This Up In Practice

### Step 1: Prepare Each Machine

All three machines need:
- Python 3.8+ installed
- The same virtual environment with:
  - `torch`, `torchvision` (PyTorch)
  - `flwr` (Flower framework)
  - Your project code (`src/segmentation/`)
- The **same model architecture file** (`model_unet.py`) — all machines must agree on the model structure

### Step 2: Prepare the Data (On Each Hospital Machine Only)

- Hospital A: Copy its X-ray images to `data/hospital_a/` on Laptop 1
- Hospital B: Copy its X-ray images to `data/hospital_b/` on Laptop 2
- Global Server: Only has the validation/test set (optional)

> [!CAUTION]
> In a real deployment, hospital data is NEVER copied between machines. Each hospital already has its own patients.

### Step 3: Start the Server First

On the Global Server laptop, run:
```
python server.py --address 0.0.0.0:8080
```

This starts listening for client connections on port 8080. The `0.0.0.0` means "accept connections from any machine on the network."

### Step 4: Start the Clients (Hospitals)

On Hospital A's laptop:
```
python client.py --server 192.168.1.100:8080 --data ./data/hospital_a/
```

On Hospital B's laptop:
```
python client.py --server 192.168.1.100:8080 --data ./data/hospital_b/
```

Each client connects to the server's IP address and waits for instructions.

### Step 5: The System Runs Automatically

Once both clients connect, the server begins the federated loop:
1. **Server** sends global model weights to both hospitals
2. **Hospitals** train locally (e.g., 1 epoch each)
3. **Hospitals** send updated weights back
4. **Server** averages the weights
5. **Repeat** for N rounds (e.g., 20 rounds)

The entire process is **automated** — no manual intervention after starting.

---

## Two Ways To Run This

### Option A: Simulation (What We Currently Have) ✅

All three "machines" run **on a single laptop** as separate processes. Flower's `start_simulation()` handles everything internally.

**Pros:** Easy to develop, debug, test. No network setup needed.
**Cons:** Not a real distributed system.

**This is what your `run_fedavg.py` and `run_fedprox.py` scripts do.**

### Option B: Real Distributed (Across Physical Machines)

Each machine runs its own process and communicates over the actual Wi-Fi network.

**Pros:** True federated learning. Proves the system works in practice.
**Cons:** Requires network setup, firewall configuration, and all machines powered on simultaneously.

### What Changes Between Simulation and Real Distributed?

| Aspect | Simulation | Real Distributed |
|--------|-----------|-----------------|
| Number of machines | 1 | 3 (or more) |
| Network | In-memory | Wi-Fi / Ethernet |
| Server startup | `start_simulation()` | `fl.server.start_server()` |
| Client startup | Handled by simulator | `fl.client.start_client()` |
| Data | Partitioned on same disk | Naturally on separate disks |
| Code changes | Minimal | Minimal |
| Model architecture | Same file | Same file on all machines |
| Training time | Sequential or parallel | Truly parallel |

> [!TIP]
> **The beauty of Flower:** Moving from simulation → real distributed requires changing only ~5 lines of code. The model, training logic, and aggregation strategy remain **identical**.

---

## The Workflow With Flower Framework — The Tool You'd Use

### Why Flower?

- **Most popular** open-source FL framework (10k+ GitHub stars)
- **Framework agnostic** — works with PyTorch, TensorFlow, JAX
- **Production-ready** — used by hospitals, banks, and telecoms
- **Simple API** — implement 4 methods and you have a working FL system

### The 4 Methods Each Hospital Implements

```
┌──────────────────────────────────────────────────────────┐
│                  Hospital Client Code                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  get_parameters()                                        │
│    → "Here are my current model weights"                 │
│    → Returns: list of NumPy arrays                       │
│                                                          │
│  set_parameters(weights)                                 │
│    → "Load these global weights into my model"           │
│    → Receives: list of NumPy arrays from server          │
│                                                          │
│  fit(weights, config)                                    │
│    → "Train on my local data with these weights"         │
│    → Receives: global weights + config (epochs, lr)      │
│    → Returns: updated weights + num_samples + metrics    │
│                                                          │
│  evaluate(weights, config)                               │
│    → "How good is this global model on my data?"         │
│    → Receives: global weights                            │
│    → Returns: loss + num_samples + metrics               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### The Server Aggregation

The server doesn't need much custom code. You just pick a **strategy**:

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **FedAvg** | `w_new = Σ(n_k/n_total) × w_k` | Standard — works when data is somewhat similar |
| **FedProx** | FedAvg + proximal penalty on clients | When data is very different (Non-IID) |

---

## Your Project: Simulation vs Reality

### What You Have Now (Simulation)

Your current `run_fedavg.py` and `run_fedprox.py` **simulate** this entire system on a single machine. Flower creates "virtual" clients internally.

```
One Laptop
├── Virtual Server (aggregates)
├── Virtual Client A (trains on client_a_train.csv)
└── Virtual Client B (trains on client_b_train.csv)
```

### What You'd Need for Real Distributed

```
Laptop 1 (Server)     Laptop 2 (Hosp A)     Laptop 3 (Hosp B)
├── server.py          ├── client.py          ├── client.py
├── model_unet.py      ├── model_unet.py      ├── model_unet.py
├── test_data/         ├── hospital_a_data/   ├── hospital_b_data/
└── results/           └── local_weights/     └── local_weights/
```

The training code, model architecture, and loss function are **100% identical** in both setups. Only the "how to connect" part changes.

---

## Security & Privacy Considerations

| Layer | What It Protects | Implementation |
|-------|-----------------|----------------|
| **Data Isolation** | Raw patient images never leave the hospital | Architecture design |
| **gRPC TLS** | Encrypts weights during transmission | Flower supports SSL/TLS certificates |
| **Differential Privacy** | Adds noise to weights to prevent reverse-engineering | Libraries like Opacus (PyTorch) |
| **Secure Aggregation** | Server can't see individual client weights | Cryptographic protocols |

For your project, the **data isolation** (weights-only communication) is the primary privacy mechanism.

---

## Summary

| Question | Answer |
|----------|--------|
| Does this system already exist? | **Yes — it's called Federated Learning** |
| What framework to use? | **Flower (flwr)** — already installed in your project |
| Can it run across real laptops on Wi-Fi? | **Yes** — Flower supports real distributed deployment via gRPC |
| What gets sent over the network? | **Only model weights** (~18 MB) — never patient data |
| How much code changes for real deployment? | **~5 lines** — swap `start_simulation()` → `start_server()` / `start_client()` |
| Is your current simulation valid? | **Yes** — scientifically equivalent to running on separate machines |
