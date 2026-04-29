import argparse
import flwr as fl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet
from segmentation.fl_client import get_parameters
from segmentation.fl_server import create_fedavg_strategy, create_fedprox_strategy

def main():
    parser = argparse.ArgumentParser("FedMedSeg Central Model Initiator (Server)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=2, help="Number of clients to wait for")
    parser.add_argument("--strategy", type=str, choices=["fedavg", "fedprox"], default="fedprox")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  FedMedSeg - Central Model Initiator (Server)")
    print(f"  Listening on: {args.host}:{args.port}")
    print(f"  Strategy: {args.strategy.upper()}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Required Clients: {args.clients}")
    print("=" * 65)

    # Initialize a dummy global model to extract initial weights
    init_model = MobileNetV2UNet(pretrained=True, freeze_encoder=False)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    if args.strategy == "fedprox":
        strategy = create_fedprox_strategy(
            proximal_mu=args.mu,
            num_clients=args.clients,
            min_fit_clients=args.clients,
            min_evaluate_clients=args.clients,
            min_available_clients=args.clients,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = create_fedavg_strategy(
            num_clients=args.clients,
            min_fit_clients=args.clients,
            min_evaluate_clients=args.clients,
            min_available_clients=args.clients,
            initial_parameters=initial_parameters,
        )

    print("\n[Server] Starting Flower Server...")
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
