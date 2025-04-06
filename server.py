import torch
import random
from trainer import Trainer

class Server:
    def __init__(self, dataset, model, args, client_selection, fed_algo):
        self.dataset = dataset
        self.model = model
        self.args = args
        self.client_selection = client_selection
        self.fed_algo = fed_algo
        self.device = args.device
        self.clients = self.create_clients()

    def create_clients(self):
        """Initialize clients with local data and trainers"""
        selected_clients = self.client_selection.select(self.args.num_clients, list(self.dataset.dataset['train']['data'].keys()))
        clients = {}

        for client_id in selected_clients:
            train_data = self.dataset.dataset['train']['data'][client_id]
            trainer = Trainer(self.model, lr=self.args.lr, epochs=self.args.epochs, device=self.device)
            clients[client_id] = {"train_data": train_data, "trainer": trainer}

        return clients

    def train(self):
        """Federated training process"""
        for round in range(self.args.epochs):
            print(f"\n🔄 Federated Round {round+1}/{self.args.epochs}")

            client_models = []
            for client_id, client in self.clients.items():
                train_loader = torch.utils.data.DataLoader(client["train_data"], batch_size=self.args.batch_size, shuffle=True)
                trained_weights = client["trainer"].train(train_loader)
                client_models.append(trained_weights)

            # Aggregate weights
            aggregated_weights = self.fed_algo.aggregate(client_models)
            self.model.load_state_dict(aggregated_weights)

        print("✅ Federated Training Completed!")
