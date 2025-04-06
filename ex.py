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

# Configure environment for stable execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.config.set_visible_devices([], 'GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuration parameters
INITIAL_TIME_INTERVAL = 25   # Initial training window in seconds
SIMILARITY_THRESHOLD = 0.8   # Cosine similarity threshold for weight pairs
GROUP_THRESHOLD = 100        # Minimum matching pairs required for grouping
MIN_CLUSTER_SIZE = 2         # Minimum clients per cluster
MIN_ROUNDS = 2               # Minimum expected rounds per client
NUM_CLIENTS = 20              # Number of clients
INITIAL_TRAINING_ROUNDS = 20  # Rounds before disconnecting from server
LOCAL_TRAINING_INTERVAL = 50  # Rounds of local training before reconnecting
CLUSTER_MERGE_INTERVAL = 30   # Interval for checking cluster merging

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
    
    Args:
        weights1: List of numpy arrays (weights of model 1).
        weights2: List of numpy arrays (weights of model 2).
        feature_importance: List of floats (importance of each feature).
    
    Returns:
        similarity: Cosine similarity between weighted weights.
    """
    # Flatten the weights
    flat_w1 = np.concatenate([w.flatten() for w in weights1])
    flat_w2 = np.concatenate([w.flatten() for w in weights2])
    
    # Apply feature importance only to the input layer weights
    input_layer_size = len(feature_importance)  # Number of input features
    weighted_w1 = flat_w1.copy()
    weighted_w2 = flat_w2.copy()
    
    # Apply feature importance to the input layer weights
    weighted_w1[:input_layer_size] *= feature_importance
    weighted_w2[:input_layer_size] *= feature_importance
    
    # Compute cosine similarity
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
    
    # Ensure minimum 50 samples per client
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

def preprocess_data(file_path):
    """Robust data preprocessing pipeline"""
    df = pd.read_csv(file_path)
    
    # Data cleaning and validation
    df = df.dropna()
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df = df[(df['fare_amount'] > 1) & (df['fare_amount'] < 500)]
    df = df[(df['pickup_longitude'].between(-180, 180)) & 
            (df['dropoff_longitude'].between(-180, 180)) &
            (df['pickup_latitude'].between(-90, 90)) & 
            (df['dropoff_latitude'].between(-90, 90))]
    df = df[df['passenger_count'].between(1, 8)]
    
    # Feature engineering
    features = ['pickup_longitude', 'pickup_latitude', 
               'dropoff_longitude', 'dropoff_latitude', 
               'passenger_count']
    X = df[features].values
    y = df['fare_amount'].values
    
    scaler = StandardScaler()
    return scaler.fit_transform(X), y

def calculate_accuracy(y_true, y_pred, tolerance=5.0):
    """Calculate accuracy as percentage of predictions within tolerance range"""
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance)
    return correct / len(y_true) * 100.0

# ========================
# Federated Trainer Class
# ========================
class FederatedTrainer:
    def __init__(self, clients_data):
        self.clients_data = clients_data
        self.base_model = self._create_compiled_model()
        self.client_weights = {name: self.base_model.get_weights() for name in clients_data}
        self.current_interval_weights = {name: [] for name in clients_data}
        self.client_weight_histories = {name: [] for name in clients_data}  # Track full history
        self.time_interval = INITIAL_TIME_INTERVAL
        self.history = []
        self.clusters = []
        self.is_connected = True
        self.cluster_matrices = {}  # Track cluster weight matrices
        self.round_counter = 0

    def _create_compiled_model(self):
        """Create and compile base model template"""
        model = Sequential([
            Input(shape=(5,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _clone_compiled_model(self):
        """Create compiled clone of base model"""
        model = clone_model(self.base_model)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_round(self, X_test, y_test):
        """Parallel client training with time-based interval"""
        start_time = time.time()
        new_weights = {}
        round_counts = {}
        self.current_interval_weights = {client: [] for client in self.clients_data}

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
        
        self.client_weights = new_weights
        self._adjust_interval(round_counts)
        self._record_history(round_counts, X_test, y_test)  # Pass X_test and y_test here
        rmse, accuracy = self.evaluate(X_test, y_test)
        print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

    def train_locally(self):
        """Train clients locally by sharing gradients within clusters"""
        print("\n=== Local Training Phase ===")
        for cluster in self.clusters:
            print(f"Training Cluster: {', '.join(cluster)}")
            cluster_weights = []
            for client in cluster:
                X, y = self.clients_data[client]
                model = self._clone_compiled_model()
                model.set_weights(self.client_weights[client])
                
                # Train locally
                model.fit(X, y, epochs=1, batch_size=16, verbose=0)
                cluster_weights.append(model.get_weights())
            
            # Average weights within cluster
            avg_weights = [np.mean(layer, axis=0) for layer in zip(*cluster_weights)]
            for client in cluster:
                self.client_weights[client] = avg_weights

    def _adjust_interval(self, round_counts):
        """Dynamically adjust training window"""
        valid_rounds = [r for r in round_counts.values() if r > 0]
        if not valid_rounds:
            return
            
        avg_rounds = np.mean(valid_rounds)
        if avg_rounds < MIN_ROUNDS:
            self.time_interval = min(self.time_interval * 1.2, 60)
        else:
            self.time_interval = max(self.time_interval * 0.9, 15)

    def _get_cluster_matrix(self, cluster):
        """Create combined weight matrix for a cluster"""
        cluster_weights = []
        for client in cluster:
            # Flatten all historical weights for this client
            client_weights = [np.concatenate([w.flatten() for w in weights]) 
                            for weights in self.client_weight_histories[client]]
            cluster_weights.extend(client_weights)
        return np.column_stack(cluster_weights) if cluster_weights else np.array([])
    
    def _calculate_cluster_similarity(self, cluster1, cluster2):
        """Calculate similarity between two clusters using their combined matrices"""
        matrix1 = self.cluster_matrices.get(tuple(cluster1), np.array([]))
        matrix2 = self.cluster_matrices.get(tuple(cluster2), np.array([]))
        
        if matrix1.size == 0 or matrix2.size == 0:
            return 0.0
        
        # Calculate similarity between all column pairs
        similarity_count = 0
        for col1 in matrix1.T:
            for col2 in matrix2.T:
                similarity = rbf_kernel([col1], [col2], gamma=0.01)
                if similarity >= SIMILARITY_THRESHOLD:
                    similarity_count += 1
        return similarity_count

    def _merge_clusters(self):
        """Check and merge clusters based on their combined matrices"""
        new_clusters = []
        merged = set()
        
        # Update cluster matrices
        self.cluster_matrices = {tuple(cluster): self._get_cluster_matrix(cluster) 
                               for cluster in self.clusters}
        
        # Check all cluster pairs
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
        
        # Add unmerged clusters
        for cluster in self.clusters:
            if tuple(cluster) not in merged:
                new_clusters.append(cluster)
        
        self.clusters = new_clusters

    def _record_history(self, round_counts, X_test, y_test):
        """Track training metrics, perform clustering, and handle cluster merging."""
        # Track weight differences from the base model
        diffs = {}
        base_weights = self.base_model.get_weights()
        flat_base = np.concatenate([w.flatten() for w in base_weights])
        
        for client in self.client_weights:
            flat_client = np.concatenate([w.flatten() for w in self.client_weights[client]])
            diffs[client] = np.linalg.norm(flat_client - flat_base)

        # Update weight histories for all clients
        for client in self.clients_data:
            self.client_weight_histories[client].extend(self.current_interval_weights[client])

        active_clients = [c for c in self.clients_data if round_counts.get(c, 0) > 0]
        similarity_counts = {}
        cluster_graph = {c: set() for c in active_clients}

        # If no clusters exist, perform initial clustering
        if not self.clusters:
            print("\n=== Performing Initial Clustering ===")
            # Calculate pairwise similarities between individual clients
            for (c1, c2) in combinations(active_clients, 2):
                count = 0
                for w1 in self.current_interval_weights[c1]:
                    for w2 in self.current_interval_weights[c2]:
                        similarity = rbf_kernel(w1, w2, gamma=0.01)  # RBF kernel
                        if similarity >= SIMILARITY_THRESHOLD:
                            count += 1
                similarity_counts[(c1, c2)] = count
                
                if count >= GROUP_THRESHOLD:
                    cluster_graph[c1].add(c2)
                    cluster_graph[c2].add(c1)

            # Find connected components for clustering
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
            
            self.clusters = clusters
        else:
            # If clusters exist, update cluster matrices and check for merging
            #print("\n=== Updating Cluster Matrices ===")
            self._update_cluster_matrices()

            # Check for cluster merging periodically
            if self.round_counter % CLUSTER_MERGE_INTERVAL == 0:
                print("\n=== Checking Cluster Merging ===")
                prev_clusters = len(self.clusters)
                self._merge_clusters()
                print(f"Merged from {prev_clusters} to {len(self.clusters)} clusters")

        # Handle new clients (if any)
        all_clients = set(self.clients_data.keys())
        clustered_clients = set().union(*self.clusters)
        new_clients = all_clients - clustered_clients

        for new_client in new_clients:
            print(f"\nEvaluating new client: {new_client}")
            max_similarity = -1
            best_cluster = None
            
            # Compare new client's matrix with existing cluster matrices
            for cluster in self.clusters:
                cluster_matrix = self.cluster_matrices.get(tuple(cluster), np.array([]))
                client_matrix = self._get_cluster_matrix([new_client])
                
                if cluster_matrix.size == 0 or client_matrix.size == 0:
                    continue
                
                similarity = self._calculate_cluster_similarity(cluster, [new_client])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = cluster
            
            if max_similarity >= GROUP_THRESHOLD:
                print(f"Adding {new_client} to cluster {self.clusters.index(best_cluster)+1}")
                best_cluster.append(new_client)
            else:
                print(f"New cluster created for {new_client}")
                self.clusters.append([new_client])

        # Print similarity report
        print(f"\n=== Interval {len(self.history)+1} Report ===")
        print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
        print(f"Grouping Threshold: {GROUP_THRESHOLD} matching pairs")
        
        for pair, count in similarity_counts.items():
            status = "PASS" if count >= GROUP_THRESHOLD else "FAIL"
            print(f"{pair[0]} vs {pair[1]}: {count} pairs ({status})")
        
        print("\nFormed Clusters:")
        if self.clusters:
            for i, cluster in enumerate(self.clusters):
                print(f"Cluster {i+1}: {', '.join(cluster)}")
        else:
            print("No clusters formed")
        print("="*50)

        # Evaluate global model and record history
        rmse,accuracy = self.evaluate(X_test, y_test)
        self.history.append({
            'round': len(self.history),
            'clusters': self.clusters,
            'diffs': diffs,
            'interval': self.time_interval,
            'similarities': similarity_counts,
            'global_rmse': rmse,  # Add global RMSE to history
            'global_accuracy' : accuracy
        })

        # Increment round counter
        self.round_counter += 1

    def aggregate_weights(self):
        """Federated averaging with fallback"""
        if not self.client_weights:
            return self.base_model.get_weights()
            
        return [np.mean(layer, axis=0) 
               for layer in zip(*self.client_weights.values())]

    def evaluate(self, X_test, y_test):
        """Robust model evaluation with both RMSE and accuracy"""
        try:
            self.base_model.set_weights(self.aggregate_weights())
            preds = self.base_model.predict(X_test, verbose=0).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            accuracy = calculate_accuracy(y_test, preds)
            return rmse, accuracy  # Return a tuple
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return float('inf'), 0.0  # Return a tuple even in error case
    
    def _update_cluster_matrices(self):
        """Update the combined weight matrices for all clusters"""
        self.cluster_matrices = {}
        for cluster in self.clusters:
            self.cluster_matrices[tuple(cluster)] = self._get_cluster_matrix(cluster)

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
    """
    Plot both RMSE and Accuracy changes after each centralized round
    """
    plt.figure(figsize=(12, 6))
    
    # Extract metrics from history
    global_rmse = [entry['global_rmse'] for entry in history]
    global_accuracy = [entry['global_accuracy'] for entry in history]
    
    # Create subplots
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
    # Load and prepare data
    train_file = 'train.csv'
    test_file = 'test.csv'
    
    X, y = preprocess_data(train_file)
    X_test, y_test = preprocess_data(test_file)
    
    # Create initial federated clients
    clients = create_clients(X, y, num_clients=NUM_CLIENTS)
    trainer = FederatedTrainer(clients)
    global_rmse = []
    global_accuracy = []  # New list for accuracy
    
    # Training loop
    while True:
        # Phase 1: Centralized training
        print("\n=== Centralized Training Phase ===")
        for round_num in range(INITIAL_TRAINING_ROUNDS):
            print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
            trainer.train_round(X_test, y_test)
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)  # Store accuracy
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
            trainer._record_history({client: 1 for client in trainer.clients_data}, X_test, y_test)
        
        # Simulate new client joining after 10 rounds
        if len(global_rmse) >= 10 and f'client_{NUM_CLIENTS+1}' not in trainer.clients_data:
            print("\n=== New Client Joined ===")
            new_client_data = create_clients(X, y, num_clients=1)
            new_client_name = list(new_client_data.keys())[0]
            trainer.clients_data[new_client_name] = new_client_data[new_client_name]
            trainer.client_weights[new_client_name] = trainer.base_model.get_weights()
            trainer.client_weight_histories[new_client_name] = []
        
        # Plot global metrics after centralized training
        plot_global_metrics(trainer.history, initial_training_rounds=INITIAL_TRAINING_ROUNDS)
        
        # Disconnect from server
        print("\n=== Disconnecting from Server ===")
        trainer.is_connected = False
        
        # Phase 2: Local training
        for local_round in range(LOCAL_TRAINING_INTERVAL):
            print(f"\n=== Local Round {local_round+1}/{LOCAL_TRAINING_INTERVAL} ===")
            trainer.train_locally()
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)  # Store accuracy
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Record history with global RMSE and accuracy
            trainer._record_history({client: 1 for client in trainer.clients_data}, X_test, y_test)
        
        # Reconnect and recluster
        print("\n=== Reconnecting to Server ===")
        trainer.is_connected = True
        trainer._record_history({client: 1 for client in trainer.clients_data}, X_test, y_test)
        
        # Check for stopping condition
        if len(global_rmse) >= 100:  # Example stopping condition
            break

    # Results analysis
    print("\nTraining Summary:")
    print(f"Best RMSE: {min(global_rmse):.4f}")
    print(f"Best Accuracy: {max(global_accuracy):.2f}%")
    print("Client Participation Summary:")
    for client in trainer.clients_data:
        active_rounds = sum(1 for entry in trainer.history 
                           if entry['diffs'][client] > 0)
        print(f"{client}: {active_rounds}/{len(trainer.history)} rounds")

    # Plot training progress and cluster evolution
    plot_training_progress(trainer.history)
    plot_cluster_evolution(trainer.history)


