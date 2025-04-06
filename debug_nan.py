import pickle
import os
import numpy as np
import torch

data_dir = "data/"  # Update with your actual data directory
file_path = os.path.join(data_dir, "FederatedTripData_preprocessed.pickle")

with open(file_path, "rb") as f:
    dataset = pickle.load(f)

# Check train data
print("Checking train dataset for NaN values...")
for client_id, tensor_dataset in dataset["train"]["data"].items():
    features, labels = tensor_dataset.tensors
    if torch.isnan(features).any() or torch.isnan(labels).any():
        print(f"⚠️ NaN found in client {client_id} training data!")

# Check test data
print("Checking test dataset for NaN values...")
for client_id, tensor_dataset in dataset["test"]["data"].items():
    features, labels = tensor_dataset.tensors
    if torch.isnan(features).any() or torch.isnan(labels).any():
        print(f"⚠️ NaN found in client {client_id} test data!")

print("✅ Debugging complete. If no warnings appear, dataset is clean.")
