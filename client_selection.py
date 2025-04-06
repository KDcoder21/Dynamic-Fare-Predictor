import numpy as np
import random

class RandomSelection:
    def __init__(self, total, device):
        self.total_clients = total
        self.device = device

    def select(self, num_selected, client_ids):
        """Randomly selects clients for training"""
        selected_clients = np.random.choice(client_ids, num_selected, replace=False)
        print(f"Selected clients: {selected_clients}")
        return selected_clients
