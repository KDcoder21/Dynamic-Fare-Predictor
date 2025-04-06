import torch

class Client:
    def __init__(self, client_id, model, train_data, test_data, optimizer, criterion, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for X, y in self.train_data:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(X).squeeze()
                loss = self.criterion(predictions, y)
                loss.backward()
                self.optimizer.step()