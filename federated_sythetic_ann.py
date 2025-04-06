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
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Update this line
from scipy.sparse import hstack, vstack, csr_matrix
from collections import defaultdict

# Configure environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuration
INITIAL_TIME_INTERVAL = 1   # Initial training window in seconds
SIMILARITY_THRESHOLD = 0.6  # Reduced from 0.8
GROUP_THRESHOLD = 20        # Reduced from 100
MIN_CLUSTER_SIZE = 1        # Allow single-client clusters initially
BATCH_SIZE = 32             # Increased from 16
BATCHES_PER_CLIENT = 10     # Added new parameter
MIN_ROUNDS = 2               # Minimum expected rounds per client
NUM_CLIENTS = 4              # Number of clients
INITIAL_TRAINING_ROUNDS = 4  # Rounds before disconnecting from server
LOCAL_TRAINING_INTERVAL = 5  # Rounds of local training before reconnecting
CLUSTER_MERGE_INTERVAL = 3   # Interval for checking cluster merging
RE_CLUSTER_INTERVAL = 10     # Interval for re-clustering in phases
INITIAL_WINDOW_SIZE = 5000  # Initial data points per client
WINDOW_EXPANSION = 50      # Data points to add per expansion
EPOCHS_PER_ROUND = 1          # Training epochs per round

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
# Memory-Efficient Encoder
# ========================
class IncrementalOneHotEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.is_fitted = False
        
    def partial_fit(self, X_categorical):
        if not self.is_fitted:
            self.encoder.fit(X_categorical)
            self.is_fitted = True
            
    def transform(self, X_categorical):
        return self.encoder.transform(X_categorical)

# ========================
# Helper Functions
# ========================
# Update the prepare_data function
def prepare_data(df, encoder=None, chunk_size=1000):
    """Incremental preprocessing with memory management"""
    df = df.dropna()
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df = df[(df['fare_amount'] > 1) & (df['fare_amount'] < 5000)]
    
    if encoder is None:
        encoder = IncrementalOneHotEncoder()
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(exclude=['object']).columns.drop('fare_amount')
    
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        
        if not categorical_cols.empty:
            encoder.partial_fit(chunk[categorical_cols])
            encoded_chunk = encoder.transform(chunk[categorical_cols])
            numeric_data = csr_matrix(chunk[numeric_cols].values)
            sparse_chunk = hstack([encoded_chunk, numeric_data])
        else:
            sparse_chunk = csr_matrix(chunk[numeric_cols].values)
            
        chunks.append(sparse_chunk.tocsr())  # Convert to CSR here
    
    X = vstack(chunks).tocsr()  # Ensure final matrix is CSR format
    y = df['fare_amount'].values
    return X, y, encoder

# Update the create_clients function
def create_clients(data, labels, num_clients=NUM_CLIENTS):
    """Create clients with contiguous data windows and dynamic max size"""
    client_names = [f'client_{i+1}' for i in range(num_clients)]
    total_samples = data.shape[0]
    samples_per_client = total_samples // num_clients
    
    clients = {}
    for i, name in enumerate(client_names):
        # Calculate client's full data window
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i != num_clients-1 else total_samples
        max_window = end_idx - start_idx
        
        # Store full data reference with dynamic max size
        clients[name] = {
            'full_data': (data[start_idx:end_idx], labels[start_idx:end_idx]),
            'window': {
                'start': 0,
                'end': min(INITIAL_WINDOW_SIZE, max_window)  # Initial window
            },
            'max_window': max_window  # Client-specific maximum
        }
        
        print(f"{name}: Total available {max_window} samples | "
              f"Initial window {clients[name]['window']['end']}")
        
    return clients

def preprocess_data(file_path, encoder=None):
    """Full preprocessing pipeline with scaling"""
    df = pd.read_csv(file_path)
    X, y, encoder = prepare_data(df, encoder)
    return X, y, encoder

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
        self.client_windows = {
            client: data['window'] for client, data in clients_data.items()
        }
        self.client_max_windows = {
            client: data['max_window'] for client, data in clients_data.items()
        }
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

    def _create_compiled_model(self):
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dense(16, activation='relu', kernel_initializer='he_normal'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5),
                    loss='mse')
        return model

    def _clone_compiled_model(self):
        model = clone_model(self.base_model)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_round(self, X_test, y_test):
        start_time = time.time()
        new_weights = {}
        round_counts = {}
        self.current_interval_weights = {client: [] for client in self.clients_data}

        for client in self.clients_data:
            try:
                X_sparse, y = self._get_client_data(client)
                if X_sparse.shape[0] < 10:
                    print(f"Skipping {client}: insufficient data")
                    continue

                model = self._clone_compiled_model()
                model.set_weights(self.client_weights[client])
                
                # Convert window to dense array once per round
                X_dense = X_sparse.toarray()
                
                # Train on entire current window
                model.fit(X_dense, y, 
                         epochs=EPOCHS_PER_ROUND,
                         batch_size=min(32, X_dense.shape[0]),
                         verbose=0)
                
                new_weights[client] = model.get_weights()
                round_counts[client] = 1  # Count as 1 round per client
                print(f"{client}: Trained on {X_dense.shape[0]} samples")   
                
            except Exception as e:
                print(f"Error training {client}: {str(e)}")
                new_weights[client] = self.client_weights[client]

        
        self.client_weights = new_weights
        self._adjust_interval(round_counts)
        self._record_history(round_counts, X_test, y_test)
        rmse, accuracy = self.evaluate(X_test, y_test)
        print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate(self, X_test, y_test, batch_size=1000):
        try:
            self.base_model.set_weights(self.aggregate_weights())
            preds = []
            valid_samples = 0
            
            for i in range(0, X_test.shape[0], batch_size):
                X_batch = X_test[i:i+batch_size].toarray()
                y_batch = y_test[i:i+batch_size]
                
                # Skip batches with NaNs/Infs
                if not np.isfinite(X_batch).all() or not np.isfinite(y_batch).all():
                    print(f"Invalid values in batch {i}-{i+batch_size}, skipping")
                    continue
                    
                batch_preds = self.base_model.predict(X_batch, verbose=0).flatten()
                preds.extend(batch_preds)
                valid_samples += len(batch_preds)

            if valid_samples == 0:
                print("No valid samples for evaluation")
                return float('inf'), 0.0
                
            # Calculate metrics only on valid samples
            rmse = np.sqrt(mean_squared_error(y_test[:valid_samples], preds))
            accuracy = calculate_accuracy(y_test[:valid_samples], preds)
            return rmse, accuracy
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return float('inf'), 0.0
    
    def _form_initial_clusters(self, active_clients):
        print("\n=== Initial Cluster Formation ===")
        client_vectors = {}
        
        # Create weight vectors for each client
        for client in active_clients:
            flat_weights = [np.concatenate([w.flatten() for w in weights]) 
                        for weights in self.current_interval_weights[client]]
            client_vectors[client] = np.mean(flat_weights, axis=0) if flat_weights else np.array([])

        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(active_clients), len(active_clients)))
        clients = list(client_vectors.keys())
        
        for i, c1 in enumerate(clients):
            for j, c2 in enumerate(clients):
                if client_vectors[c1].size == 0 or client_vectors[c2].size == 0:
                    similarity = 0.0
                else:
                    similarity = cosine_similarity(
                        [client_vectors[c1]], 
                        [client_vectors[c2]]
                    )[0][0]
                similarity_matrix[i,j] = similarity

        # Use hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        cluster_model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=1 - SIMILARITY_THRESHOLD
        )
        cluster_labels = cluster_model.fit_predict(1 - similarity_matrix)  # Convert to distance matrix
        
        # Form clusters
        clusters = defaultdict(list)
        for client_idx, cluster_id in enumerate(cluster_labels):
            clusters[cluster_id].append(clients[client_idx])
            
        return [c for c in clusters.values() if len(c) >= MIN_CLUSTER_SIZE]

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

    def _expand_client_windows(self):
        """Expand data windows for all clients"""
        print("\n=== Expanding Client Data Windows ===")
        for client in self.clients_data:
            window = self.client_windows[client]
            current_max = self.client_max_windows[client]
            
            new_end = min(
                window['end'] + WINDOW_EXPANSION,
                current_max  # Can't exceed client's total capacity
            )
            
            if new_end > window['end']:
                window['end'] = new_end
                print(f"{client}: Window expanded to {window['start']}-{window['end']} "
                      f"({window['end'] - window['start']} samples)")
            else:
                print(f"{client}: Maximum window reached ({current_max} samples)")
    
    def _get_client_data(self, client_name):
        """Get current window data for a client"""
        client_entry = self.clients_data[client_name]
        full_data, full_labels = client_entry['full_data']
        window = client_entry['window']
        
        # Convert sparse matrices to proper format
        X_window = full_data[window['start']:window['end']].tocsr()
        y_window = full_labels[window['start']:window['end']]
        
        return X_window, y_window

    def _record_history(self, round_counts, X_test, y_test):
        diffs = {}
        base_weights = self.base_model.get_weights()
        flat_base = np.concatenate([w.flatten() for w in base_weights])
        
        for client in self.client_weights:
            flat_client = np.concatenate([w.flatten() for w in self.client_weights[client]])
            diffs[client] = np.linalg.norm(flat_client - flat_base)

        active_clients = [c for c in self.clients_data if round_counts.get(c, 0) > 0]

        if not self.final_clusters:
            if not self.clusters:
                self.clusters = self._form_initial_clusters(active_clients)
            else:
                self._update_cluster_matrices()
        else:
            self._update_cluster_matrices()
            self._handle_new_clients()

        print(f"\n=== Interval {len(self.history)+1} Report ===")
        print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
        print(f"Grouping Threshold: {GROUP_THRESHOLD} matching pairs")
        
        print("\nFormed Clusters:")
        if self.clusters:
            for i, cluster in enumerate(self.clusters):
                print(f"Cluster {i+1}: {', '.join(cluster)}")
        else:
            print("No clusters formed")
        print("="*50)

        rmse, accuracy = self.evaluate(X_test, y_test)
        self.history.append({
            'round': len(self.history),
            'clusters': self.clusters,
            'diffs': diffs,
            'interval': self.time_interval,
            'global_rmse': rmse,
            'global_accuracy': accuracy
        })

        self.round_counter += 1

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
    train_file = '/mnt/E_shared/project/train_dataset.csv'
    test_file = '/mnt/E_shared/project/test_dataset.csv'
    
    encoder = None
    X_train, y_train, encoder = preprocess_data(train_file)
    print(X_train.shape[0])
    X_test, y_test, _ = preprocess_data(test_file,encoder)
    print(X_test.shape[0])

    total_samples = X_train.shape[0]
    print(f"\nTotal training samples: {total_samples}")
    print(f"Samples per client: ~{total_samples // NUM_CLIENTS}")
    
    clients = create_clients(X_train, y_train, num_clients=NUM_CLIENTS)
    trainer = FederatedTrainer(clients)
    global_rmse = []
    global_accuracy = []
    
    print("\n=== Phase 1: Initial Centralized Phase ===")
    for round_num in range(INITIAL_TRAINING_ROUNDS):
        print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
        trainer.train_round(X_test, y_test)
    trainer._finalize_clusters()
    trainer.phase_counter = 1

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