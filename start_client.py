import argparse
import flwr as fl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet
from segmentation.dataset_rsna import RSNAPneumoniaDataset
from segmentation.fl_client import FedMedSegClient

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def main():
    parser = argparse.ArgumentParser("FedMedSeg Hospital Node (Client)")
    parser.add_argument("--server", type=str, required=True, help="Server IP address and port (e.g., 192.168.1.100:8080)")
    parser.add_argument("--node-type", type=str, choices=["client_a", "client_b"], required=True, help="Which dataset to use (client_a or client_b)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    
    # Strategy args (should match server expectations)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient (0.0 for FedAvg)")
    
    # DP Args
    parser.add_argument("--use-dp", action="store_true", help="Enable Differential Privacy")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target Epsilon for DP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    
    args = parser.parse_args()

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("\n" + "=" * 65)
    print(f"  FedMedSeg - Hospital Node ({args.node_type})")
    print(f"  Connecting to server: {args.server}")
    print(f"  Device: {device}")
    print(f"  DP Enabled: {args.use_dp}")
    if args.mu > 0:
        print(f"  Strategy: FedProx (mu={args.mu})")
    else:
        print(f"  Strategy: FedAvg")
    print("=" * 65)

    rsna_root = str(PROJECT_ROOT / "data" / "rsna_pneumonia")
    train_csv = str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / f"{args.node_type}_train.csv")
    val_csv   = str(PROJECT_ROOT / "data" / "rsna_pneumonia" / "subset" / "val_subset.csv")

    if not Path(train_csv).exists():
        print(f"Error: Could not find dataset {train_csv}")
        sys.exit(1)

    print("\n[Client] Loading datasets...")
    train_ds = RSNAPneumoniaDataset(
        rsna_root=rsna_root,
        subset_csv=train_csv,
        img_transform=get_train_transform(),
        augment=True,
    )
    val_ds = RSNAPneumoniaDataset(
        rsna_root=rsna_root,
        subset_csv=val_csv,
        img_transform=get_val_transform(),
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("[Client] Initializing model...")
    model = MobileNetV2UNet(pretrained=False, freeze_encoder=False).to(device)

    client = FedMedSegClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        client_name=args.node_type,
        local_epochs=args.local_epochs,
        lr=1e-4,
        mu=args.mu,
        use_dp=args.use_dp,
        target_epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
    )

    print(f"\n[Client] Connecting to server {args.server}...")
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
