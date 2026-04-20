"""
Minimal diagnostic script to identify WHY Opacus per-sample gradients
are not being computed for MobileNetV2UNet.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from segmentation.model_unet import MobileNetV2UNet
from segmentation.loss import DiceBCELoss
from segmentation.fl_client import prepare_model_for_dp

# ── Step 1: Create and prepare model ─────────────────────────────────────────
print("=" * 60)
print("OPACUS DIAGNOSTIC TEST")
print("=" * 60)

model = MobileNetV2UNet(pretrained=False, freeze_encoder=False)
print(f"\n[1] Model created. Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"    Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Prepare for DP
model = prepare_model_for_dp(model)
print(f"\n[2] Model prepared for DP (GroupNorm + no-inplace)")
print(f"    Trainable params after fix: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Check validity
from opacus.validators import ModuleValidator
is_valid = ModuleValidator.is_valid(model)
print(f"    ModuleValidator.is_valid: {is_valid}")

if not is_valid:
    errors = ModuleValidator.validate(model)
    print(f"    Validation errors:")
    for err in errors:
        print(f"      - {err}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[3] Device: {device}")
model = model.to(device)

# ── Step 2: Create dummy data ────────────────────────────────────────────────
images = torch.randn(8, 3, 224, 224)
masks = torch.randint(0, 2, (8, 1, 224, 224)).float()
dataset = TensorDataset(images, masks)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ── Step 3: Create optimizer ─────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print(f"\n[4] Optimizer created with {len(list(model.parameters()))} parameter groups")

# ── Step 4: Wrap with Opacus ─────────────────────────────────────────────────
from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()

try:
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        epochs=1,
    )
    print(f"\n[5] PrivacyEngine wrapped successfully")
    print(f"    Model type: {type(model).__name__}")
    print(f"    Optimizer type: {type(optimizer).__name__}")
except Exception as e:
    print(f"\n[5] PrivacyEngine FAILED: {e}")
    sys.exit(1)

# ── Step 5: Test forward + backward ──────────────────────────────────────────
criterion = DiceBCELoss(smooth=1e-6)
model.train()

for images_batch, masks_batch in loader:
    images_batch = images_batch.to(device)
    masks_batch = masks_batch.to(device)

    optimizer.zero_grad()
    
    print(f"\n[6] Running forward pass...")
    preds = model(images_batch)
    print(f"    Preds shape: {preds.shape}, range: [{preds.min().item():.4f}, {preds.max().item():.4f}]")
    
    loss = criterion(preds, masks_batch)
    print(f"    Loss: {loss.item():.4f}")
    
    print(f"\n[7] Running backward pass...")
    loss.backward()
    
    # ── Check grad_sample on ALL parameters ──────────────────────────────
    has_grad_sample = []
    missing_grad_sample = []
    has_grad = []
    missing_grad = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if hasattr(param, 'grad_sample') and param.grad_sample is not None:
            has_grad_sample.append(name)
        else:
            missing_grad_sample.append(name)
        
        if param.grad is not None:
            has_grad.append(name)
        else:
            missing_grad.append(name)
    
    print(f"\n[8] GRADIENT ANALYSIS:")
    print(f"    Params with grad_sample:    {len(has_grad_sample)}")
    print(f"    Params WITHOUT grad_sample: {len(missing_grad_sample)}")
    print(f"    Params with .grad:          {len(has_grad)}")
    print(f"    Params WITHOUT .grad:       {len(missing_grad)}")
    
    if missing_grad_sample:
        print(f"\n    MISSING grad_sample on these params:")
        for name in missing_grad_sample[:20]:
            param = dict(model.named_parameters())[name]
            grad_info = f"grad={'YES' if param.grad is not None else 'NO'}"
            print(f"      - {name}  shape={list(param.shape)}  {grad_info}")
        if len(missing_grad_sample) > 20:
            print(f"      ... and {len(missing_grad_sample) - 20} more")
    
    # ── Try optimizer.step() ─────────────────────────────────────────────
    print(f"\n[9] Attempting optimizer.step()...")
    try:
        optimizer.step()
        print(f"    SUCCESS!")
    except ValueError as e:
        print(f"    FAILED: {e}")
    
    break

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
