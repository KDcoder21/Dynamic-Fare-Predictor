import torch
import argparse
from model import build_model
from server import Server
from Taxi_fare.federated_synthetic import FederatedTripDataDataset
from client_selection import RandomSelection
from federated_algorithm import FedAvg

# ✅ Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/", help="Dataset directory")
parser.add_argument("--num_clients", type=int, default=62, help="Total clients")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Local training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
args = parser.parse_args()

# ✅ Load dataset
dataset = FederatedTripDataDataset(args.data_dir, args)
args.input_dim = dataset.input_dim  # Set input feature dimension dynamically
print(f"Total Clients: {args.num_clients}, Input Dimension: {args.input_dim}")

# ✅ Initialize model
model = build_model(input_dim=args.input_dim).to(args.device)

# ✅ Client Selection
client_selection = RandomSelection(total=args.num_clients, device=args.device)

# ✅ Federated Algorithm
fed_algo = FedAvg(dataset.train_data_sizes, model)

# ✅ Start Federated Training
server = Server(dataset, model, args, client_selection, fed_algo)
server.train()

#python main.py --data_dir data/ --num_clients 62 --batch_size 32 --epochs 10 --lr 0.0005

