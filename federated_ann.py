import numpy as np
import pandas as pd
import time
import os
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configure environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuration
INITIAL_TIME_INTERVAL = 25   # Initial training window in seconds
SIMILARITY_THRESHOLD = 0.8   # Cosine similarity threshold for weight pairs
GROUP_THRESHOLD = 100        # Minimum matching pairs required for grouping
MIN_CLUSTER_SIZE = 1         # Minimum clients per cluster
MIN_ROUNDS = 2               # Minimum expected rounds per client
NUM_CLIENTS = 4              # Number of clients
INITIAL_TRAINING_ROUNDS = 4  # Rounds before disconnecting from server
LOCAL_TRAINING_INTERVAL = 5  # Rounds of local training before reconnecting
CLUSTER_MERGE_INTERVAL = 3   # Interval for checking cluster merging
RE_CLUSTER_INTERVAL = 10     # Interval for re-clustering in phases

# ========================
# Similarity Measures
# ========================
def cosine_weight_similarity(weights1, weights2):
    """Compute cosine similarity between flattened weights"""
    flat_w1 = np.concatenate([w.flatten() for w in weights1])
    flat_w2 = np.concatenate([w.flatten() for w in weights2])
    return cosine_similarity([flat_w1], [flat_w2])[0][0]

def layer_wise_weighted_similarity(weights1, weights2, layer_weights=[0.4, 0.3, 0.3]):
    """Compute weighted similarity across layers"""
    similarities = []
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        flat_w1 = w1.flatten()
        flat_w2 = w2.flatten()
        similarity = cosine_similarity([flat_w1], [flat_w2])[0][0]
        similarities.append(similarity * layer_weights[i])
    return np.sum(similarities)

def cka_similarity(weights1, weights2):
    """Compute Centered Kernel Alignment (CKA) similarity"""
    X = np.concatenate([w.flatten() for w in weights1]).reshape(1, -1)
    Y = np.concatenate([w.flatten() for w in weights2]).reshape(1, -1)
    
    # Center the features
    X_centered = X - X.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)
    
    # Compute CKA
    XYT = X_centered.dot(Y_centered.T)
    return (np.trace(XYT) / (np.linalg.norm(X_centered, 'fro') * np.linalg.norm(Y_centered, 'fro'))) ** 2

def importance_weighted_similarity(weights1, weights2, feature_importance):
    """
    Compute weighted similarity between weights based on feature importance.
    Only applies feature importance to the input layer weights.
    """
    # Flatten the weights
    flat_w1 = np.concatenate([w.flatten() for w in weights1])
    flat_w2 = np.concatenate([w.flatten() for w in weights2])
    
    # Apply feature importance only to the input layer weights
    input_layer_size = len(feature_importance)
    weighted_w1 = flat_w1.copy()
    weighted_w2 = flat_w2.copy()
    
    weighted_w1[:input_layer_size] *= feature_importance
    weighted_w2[:input_layer_size] *= feature_importance
    
    return cosine_similarity([weighted_w1], [weighted_w2])[0][0]

def temporal_similarity(weights1, weights2, time_feature_indices):
    """Compare weights handling temporal features"""
    time_weights1 = [w[time_feature_indices] for w in weights1 if w.ndim > 1]
    time_weights2 = [w[time_feature_indices] for w in weights2 if w.ndim > 1]
    return cosine_similarity(
        np.concatenate(time_weights1).flatten(),
        np.concatenate(time_weights2).flatten()
    )

def rbf_kernel(weights1, weights2, gamma=0.1):
    """Radial Basis Function (RBF) kernel"""
    flat_w1 = np.concatenate([w.flatten() for w in weights1])
    flat_w2 = np.concatenate([w.flatten() for w in weights2])
    distance = np.linalg.norm(flat_w1 - flat_w2) ** 2
    return np.exp(-gamma * distance)

def polynomial_kernel(weights1, weights2, degree=3, c=1):
    """Polynomial kernel"""
    flat_w1 = np.concatenate([w.flatten() for w in weights1])
    flat_w2 = np.concatenate([w.flatten() for w in weights2])
    return (np.dot(flat_w1, flat_w2) + c) ** degree

def laplacian_kernel(weights1, weights2, gamma=0.1):
    """Laplacian kernel"""
    flat_w1 = np.concatenate([w.flatten() for w in weights1])
    flat_w2 = np.concatenate([w.flatten() for w in weights2])
    distance = np.linalg.norm(flat_w1 - flat_w2, ord=1)
    return np.exp(-gamma * distance)

# ========================
# Helper Functions
# ========================
def create_clients(data, labels, num_clients=NUM_CLIENTS):
    """Create balanced client shards with validation"""
    client_names = [f'client_{i+1}' for i in range(num_clients)]
    data, labels = shuffle(data, labels, random_state=42)
    
    min_samples = max(50, len(data) // num_clients)
    shards = []
    start = 0
    for _ in range(num_clients):
        end = min(start + min_samples, len(data))
        shards.append((data[start:end], labels[start:end]))
        start = end
        if start >= len(data):
            break
            
    return {client_names[i]: shards[i] for i in range(len(shards))}

def prepare_data(df, encoder=None, scaler=None, fit=True):
    """Preprocess data with proper feature engineering"""
    # Clean data
    df = df.dropna()
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df = df[(df['fare_amount'] > 1) & (df['fare_amount'] < 5000)]
    df = df.reset_index(drop=True)

    # Feature processing
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']

    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    if fit or encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Changed here
    if not categorical_cols.empty:
        encoded_data = encoder.fit_transform(X[categorical_cols]) if fit else encoder.transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        X = pd.concat([X.drop(categorical_cols, axis=1), encoded_df], axis=1)

    # Scale numerical features
    if fit or scaler is None:
        scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols]) if fit else scaler.transform(X[numeric_cols])

    return X.values.astype(np.float32), y.values, encoder, scaler

def preprocess_data(train_path, test_path):
    """Consistent preprocessing for both train and test"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Process training data
    X_train, y_train, encoder, scaler = prepare_data(train_df, fit=True)
    
    # Process test data with same transformers
    X_test, y_test, _, _ = prepare_data(test_df, encoder=encoder, scaler=scaler, fit=False)
    
    return X_train, y_train, X_test, y_test

def calculate_accuracy(y_true, y_pred, tolerance=5.0):
    """Calculate accuracy as percentage of predictions within tolerance range"""
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance)
    return correct / len(y_true) * 100.0

# ========================
# Federated Trainer Class
# ========================
class FederatedTrainer:
    def __init__(self, clients_data, input_dim):
        self.input_dim = input_dim
        self.clients_data = clients_data
        self.base_model = self._create_compiled_model()
        self.client_weights = {name: self.base_model.get_weights() for name in clients_data}
        self.current_interval_weights = {name: [] for name in clients_data}
        self.client_weight_histories = {name: [] for name in clients_data}
        self.time_interval = INITIAL_TIME_INTERVAL
        self.history = []
        self.clusters = []
        self.final_clusters = None
        self.cluster_matrices = {}
        self.round_counter = 0
        self.phase_counter = 0
        self.global_round_count = 0
        

    def _create_compiled_model(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),  # Use provided input dimension
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _clone_compiled_model(self):
        model = clone_model(self.base_model)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def federated_average(self, client_weights_list):
        """FedAvg implementation: average weights"""
        avg_weights = []
        for layer in zip(*client_weights_list):
            avg_layer = np.mean(layer, axis=0)
            avg_weights.append(avg_layer)
        return avg_weights

    def aggregate_weights(self, aggregation_algo='fedavg'):
        """Aggregation handler with algorithm selection"""
        weight_list = list(self.client_weights.values())
        
        if aggregation_algo == 'fedavg':
            return self.federated_average(weight_list)
        # Add other aggregation methods here
        else:
            raise ValueError(f"Unknown aggregation algorithm: {aggregation_algo}")

    def train_round(self, X_test, y_test):
        start_time = time.time()
        new_weights = {}
        round_counts = {}
        self.current_interval_weights = {client: [] for client in self.clients_data}

        # Local training phase
        for client, (X, y) in self.clients_data.items():
            try:
                if len(X) < 10 or len(y) < 10:
                    print(f"Skipping {client}: insufficient data")
                    continue
                
                model = self._clone_compiled_model()
                model.set_weights(self.client_weights[client])
                
                time_per_client = max(1, self.time_interval / len(self.clients_data))
                client_start = time.time()
                rounds = 0
                
                while time.time() - client_start < time_per_client:
                    batch_size = min(16, max(8, len(X)//2))
                    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)
                    self.current_interval_weights[client].append(model.get_weights())
                    rounds += 1
                
                new_weights[client] = model.get_weights()
                round_counts[client] = rounds
                print(f"{client}: {rounds} rounds")   
            except Exception as e:
                print(f"Error training {client}: {str(e)}")
                new_weights[client] = self.client_weights[client]

        # Weight aggregation
        aggregated_weights = self.aggregate_weights()
        for client in self.clients_data:
            self.client_weights[client] = aggregated_weights
            self.client_weight_histories[client].append(aggregated_weights)

        self._adjust_interval(round_counts)
        self._record_history(round_counts, X_test, y_test)
        
        # Only cluster after final global round
        self.global_round_count += 1
        if self.global_round_count == INITIAL_TRAINING_ROUNDS:
            self._perform_final_clustering()
            
        rmse, accuracy = self.evaluate(X_test, y_test)
        print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate(self, X_test, y_test):
        try:
            self.base_model.set_weights(self.aggregate_weights())
            preds = self.base_model.predict(X_test, verbose=0).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            accuracy = calculate_accuracy(y_test, preds)
            return rmse, accuracy
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return float('inf'), 0.0
    
    def _form_initial_clusters(self, active_clients):
        print("\n=== Initial Cluster Formation ===")
        cluster_graph = {c: set() for c in active_clients}
        similarity_counts = {}
        
        for (c1, c2) in combinations(active_clients, 2):
            count = 0
            for w1 in self.current_interval_weights[c1]:
                for w2 in self.current_interval_weights[c2]:
                    similarity = rbf_kernel(w1, w2, gamma=0.01)
                    if similarity >= SIMILARITY_THRESHOLD:
                        count += 1
            similarity_counts[(c1, c2)] = count
            if count >= GROUP_THRESHOLD:
                cluster_graph[c1].add(c2)
                cluster_graph[c2].add(c1)

        if self.phase_counter % RE_CLUSTER_INTERVAL == 0:
            print("\n=== Similarity Pair Logging ===")
            for pair, count in similarity_counts.items():
                status = "PASS" if count >= GROUP_THRESHOLD else "FAIL"
                print(f"{pair[0]} vs {pair[1]}: {count} pairs ({status})")

        visited = set()
        clusters = []
        for client in active_clients:
            if client not in visited and len(cluster_graph[client]) > 0:
                cluster = set()
                queue = [client]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        queue.extend([n for n in cluster_graph[current] if n not in visited])
                if len(cluster) >= MIN_CLUSTER_SIZE:
                    clusters.append(cluster)
        return clusters

    def _finalize_clusters(self):
        print("\n=== Final Cluster Merging ===")
        self.cluster_matrices = {tuple(cluster): self._get_cluster_matrix(cluster) 
                               for cluster in self.clusters}
        
        new_clusters = []
        merged = set()
        for i, cluster1 in enumerate(self.clusters):
            if tuple(cluster1) in merged:
                continue
                
            current_cluster = set(cluster1)
            for j, cluster2 in enumerate(self.clusters[i+1:]):
                similarity = self._calculate_cluster_similarity(cluster1, cluster2)
                if similarity >= GROUP_THRESHOLD:
                    current_cluster.update(cluster2)
                    merged.add(tuple(cluster2))
            
            new_clusters.append(list(current_cluster))
            merged.add(tuple(cluster1))
        
        print(f"Merged from {len(self.clusters)} to {len(new_clusters)} clusters")
        self.final_clusters = new_clusters
        self.clusters = new_clusters
        self._update_cluster_matrices()

    def _perform_final_clustering(self):
        """Perform clustering only once after final global round"""
        print("\n=== Final Clustering Phase ===")
        active_clients = [c for c in self.clients_data]
        self.clusters = self._form_initial_clusters(active_clients)
        self._finalize_clusters()

    def _record_history(self, round_counts, X_test, y_test):
        """Modified to remove clustering from history recording"""
        diffs = {}
        base_weights = self.base_model.get_weights()
        flat_base = np.concatenate([w.flatten() for w in base_weights])
        
        for client in self.client_weights:
            flat_client = np.concatenate([w.flatten() for w in self.client_weights[client]])
            diffs[client] = np.linalg.norm(flat_client - flat_base)

        self.history.append({
            'round': len(self.history),
            'clusters': self.clusters.copy(),
            'diffs': diffs.copy(),
            'interval': self.time_interval,
            'global_rmse': self.evaluate(X_test, y_test)[0],
            'global_accuracy': self.evaluate(X_test, y_test)[1]
        })

    def _handle_new_clients(self):
        all_clients = set(self.clients_data.keys())
        clustered_clients = set().union(*self.final_clusters)
        new_clients = all_clients - clustered_clients

        for new_client in new_clients:
            print(f"\nEvaluating new client: {new_client}")
            max_similarity = -1
            best_cluster = None
            
            for cluster in self.final_clusters:
                cluster_matrix = self.cluster_matrices.get(tuple(cluster), np.array([]))
                client_matrix = self._get_cluster_matrix([new_client])
                
                if cluster_matrix.size == 0 or client_matrix.size == 0:
                    continue
                
                similarity = self._calculate_cluster_similarity(cluster, [new_client])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = cluster
            
            if max_similarity >= GROUP_THRESHOLD:
                print(f"Adding {new_client} to cluster {self.final_clusters.index(best_cluster)+1}")
                best_cluster.append(new_client)
            else:
                print(f"New cluster created for {new_client}")
                self.final_clusters.append([new_client])

    def _calculate_cluster_similarity(self, cluster1, cluster2):
        matrix1 = self.cluster_matrices.get(tuple(cluster1), np.array([]))
        matrix2 = self.cluster_matrices.get(tuple(cluster2), np.array([]))
        
        if matrix1.size == 0 or matrix2.size == 0:
            return 0.0
        
        similarity_count = 0
        for col1 in matrix1.T:
            for col2 in matrix2.T:
                similarity = rbf_kernel([col1], [col2], gamma=0.01)
                if similarity >= SIMILARITY_THRESHOLD:
                    similarity_count += 1
        return similarity_count

    def _get_cluster_matrix(self, cluster):
        cluster_weights = []
        for client in cluster:
            client_weights = [np.concatenate([w.flatten() for w in weights]) 
                            for weights in self.client_weight_histories[client]]
            cluster_weights.extend(client_weights)
        return np.column_stack(cluster_weights) if cluster_weights else np.array([])

    def _update_cluster_matrices(self):
        self.cluster_matrices = {tuple(cluster): self._get_cluster_matrix(cluster) 
                               for cluster in self.clusters}

    def train_locally(self):
        print("\n=== Local Training Phase ===")
        for cluster in self.final_clusters:
            print(f"Training Cluster: {', '.join(cluster)}")
            cluster_weights = []
            for client in cluster:
                X, y = self.clients_data[client]
                model = self._clone_compiled_model()
                model.set_weights(self.client_weights[client])
                
                model.fit(X, y, epochs=1, batch_size=16, verbose=0)
                cluster_weights.append(model.get_weights())
            
            avg_weights = [np.mean(layer, axis=0) for layer in zip(*cluster_weights)]
            for client in cluster:
                self.client_weights[client] = avg_weights

    def _adjust_interval(self, round_counts):
        valid_rounds = [r for r in round_counts.values() if r > 0]
        if not valid_rounds:
            return
            
        avg_rounds = np.mean(valid_rounds)
        if avg_rounds < MIN_ROUNDS:
            self.time_interval = min(self.time_interval * 1.2, 60)
        else:
            self.time_interval = max(self.time_interval * 0.9, 15)

    def re_cluster(self):
        print("\n=== Re-clustering Process ===")
        self.clusters = []
        self.final_clusters = None
        self.cluster_matrices = {}
        
        active_clients = list(self.clients_data.keys())
        
        self.clusters = self._form_initial_clusters(active_clients)
        self._finalize_clusters()
        
        print("Re-clustering completed. New clusters:")
        for i, cluster in enumerate(self.final_clusters):
            print(f"Cluster {i+1}: {', '.join(cluster)}")

    def aggregate_weights(self):
        if not self.client_weights:
            return self.base_model.get_weights()
        return [np.mean(layer, axis=0) for layer in zip(*self.client_weights.values())]

# ========================
# Visualization Functions
# ========================
def plot_training_progress(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    active_counts = [len([c for c in entry['diffs'] if entry['diffs'][c] > 0]) 
                    for entry in history]
    plt.plot(active_counts, marker='o')
    plt.title('Active Clients per Round')
    plt.xlabel('Round')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    intervals = [entry['interval'] for entry in history]
    plt.plot(intervals, marker='s', color='orange')
    plt.title('Training Interval Adjustment')
    plt.xlabel('Round')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    for client in history[0]['diffs']:
        diffs = [entry['diffs'].get(client, 0) for entry in history]
        plt.plot(diffs, alpha=0.5, label=client)
    plt.title('Model Weight Changes')
    plt.xlabel('Round')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cluster_evolution(history):
    plt.figure(figsize=(12, 6))
    all_clients = list(history[0]['diffs'].keys())
    num_rounds = len(history)
    cluster_matrix = np.zeros((len(all_clients), num_rounds))
    
    for round_idx, entry in enumerate(history):
        for cluster_idx, cluster in enumerate(entry['clusters']):
            for client in cluster:
                client_idx = all_clients.index(client)
                cluster_matrix[client_idx, round_idx] = cluster_idx + 1

    plt.imshow(cluster_matrix, aspect='auto', cmap='tab20', vmin=0, vmax=len(history[0]['clusters'])+1)
    plt.colorbar(label='Cluster ID', ticks=range(len(history[0]['clusters'])+2))
    plt.yticks(range(len(all_clients)), all_clients)
    plt.xlabel("Communication Round")
    plt.ylabel("Client")
    plt.title("Cluster Membership Evolution")
    plt.show()

def plot_global_metrics(history, initial_training_rounds=INITIAL_TRAINING_ROUNDS):
    plt.figure(figsize=(12, 6))
    
    global_rmse = [entry['global_rmse'] for entry in history]
    global_accuracy = [entry['global_accuracy'] for entry in history]
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(global_rmse) + 1), global_rmse, marker='o', color='b', label="RMSE")
    plt.axvline(x=initial_training_rounds, color='r', linestyle='--', label="Centralized Training End")
    plt.title('Global RMSE')
    plt.xlabel('Rounds')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(global_accuracy) + 1), global_accuracy, marker='o', color='g', label="Accuracy")
    plt.axvline(x=initial_training_rounds, color='r', linestyle='--', label="Centralized Training End")
    plt.title('Accuracy (±$5)')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    train_file = '/mnt/E_shared/project/train_dataset_10000.csv'
    test_file = '/mnt/E_shared/project/test_dataset_2000.csv'
    
    X_train, y_train, X_test, y_test = preprocess_data(train_file, test_file)
    
    clients = create_clients(X_train, y_train, num_clients=NUM_CLIENTS)
    trainer = FederatedTrainer(clients,input_dim=X_train.shape[1])
    global_rmse = []
    global_accuracy = []
    
    print("\n=== Phase 1: Initial Centralized Phase ===")
    for round_num in range(INITIAL_TRAINING_ROUNDS):
        print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
        trainer.train_round(X_test, y_test)
    print("\n=== Final Cluster Results ===")
    for i, cluster in enumerate(trainer.final_clusters):
        print(f"Cluster {i+1}: {', '.join(cluster)}")
    trainer.phase_counter += 1

    while True:
        print(f"\n=== Phase {trainer.phase_counter+1}: Local Training ===")
        for local_round in range(LOCAL_TRAINING_INTERVAL):
            print(f"\n=== Local Round {local_round+1}/{LOCAL_TRAINING_INTERVAL} ===")
            trainer.train_locally()
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
        trainer.phase_counter += 1
        
        if trainer.phase_counter % RE_CLUSTER_INTERVAL == 0:
            trainer.re_cluster()

        print(f"\n=== Phase {trainer.phase_counter+1}: Centralized Training ===")
        for round_num in range(INITIAL_TRAINING_ROUNDS):
            print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
            trainer.train_round(X_test, y_test)
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
        trainer.phase_counter += 1

        if trainer.phase_counter % RE_CLUSTER_INTERVAL == 0:
            trainer.re_cluster()

        trainer._handle_new_clients()
        
        if len(global_rmse) >= 100:
            break

    print("\nTraining Summary:")
    print(f"Best RMSE: {min(global_rmse):.4f}")
    print(f"Best Accuracy: {max(global_accuracy):.2f}%")
    print("Client Participation Summary:")
    for client in trainer.clients_data:
        active_rounds = sum(1 for entry in trainer.history 
                           if entry['diffs'][client] > 0)
        print(f"{client}: {active_rounds}/{len(trainer.history)} rounds")

    plot_training_progress(trainer.history)
    plot_cluster_evolution(trainer.history)
    plot_global_metrics(trainer.history, initial_training_rounds=INITIAL_TRAINING_ROUNDS)