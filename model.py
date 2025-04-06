import torch
import torch.nn as nn
import torch.optim as optim

class TaxiFareNN(nn.Module):
    def __init__(self, input_dim=20):  # 20 input features
        super(TaxiFareNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 48)
        self.fc2 = nn.Linear(48, 96)
        self.fc3 = nn.Linear(96, 48)
        self.fc4 = nn.Linear(48, 24)
        self.fc5 = nn.Linear(24, 12)
        self.fc6 = nn.Linear(12, 6)
        self.fc7 = nn.Linear(6, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)  # No activation (for regression)
        return x

def build_model(input_dim=20):
    model = TaxiFareNN(input_dim)
    return model
