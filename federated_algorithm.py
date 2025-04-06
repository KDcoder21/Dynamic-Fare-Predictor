import torch

class FedAvg:
    def __init__(self, client_data_sizes, model):
        self.client_data_sizes = client_data_sizes
        self.global_model = model

    def aggregate(self, client_updates):
        """Federated averaging"""
        total_samples = sum(self.client_data_sizes[c] for c in range(len(client_updates)))
        new_state_dict = self.global_model.state_dict()

        for key in new_state_dict.keys():
            new_state_dict[key] = sum(
                client_updates[i][key] * self.client_data_sizes[i] / total_samples
                for i in range(len(client_updates))
            )

        return new_state_dict
