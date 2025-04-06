import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, lr=0.001, epochs=10, device="cpu"):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_loader):
        """Train model on local data"""
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features).squeeze()
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self.model.state_dict()
