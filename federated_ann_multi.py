#For eg, there are 4 clients at start each getting 1/4th of dataset for training and testing. first global phase starts which has INITIAL_TRAINING_ROUNDS = 4 rounds in round1 of phase1, each client trains for some time interval and shares the latest weight with the server and server aggregates this weight and sends back to all clients and same happens in round 2,3 but in round4 we collect all weights of all rounds of all clients trained in this round and send to server because this weight vector is useful for clustering. So, server does clustering in this round and clusters are formed and move to phase2(local phase). Clients train in local round then phase3(global phase). In this phase, write a seperate function to select client from each cluster in random(this algo will be changed in future, so if you write it as a seperate function I will change in future) then the same happens for round 1,2 and 3 ,from each we select a client a random and do the weight aggregation and send back to all clients present. In round4 of this phase, I want to add a new client with it's data sourced from same dataset and as same size of other clients but selected in random and in this phase we will train all clients for that same time interval, but old clients which were in cluster sends merged matrices to server and this new server sends a vector of weights.(here since there are 4 clients new client will be client5 let client1 and 3 formed cluster and client2 and 4 formed another cluster, in this round let client1 trained 3 times generating a matrix of size 3*number_of_weights and client3 trained 4 times generating a matrix of size 4*number_of_weights similarly,client2 trained 5 times generating a matrix of size 5*number_of_weights and client4 trained 6 times generating a matrix of size 6*number_of_weights and client5 trained 2 times generating a matrix of size 2*number_of_weights. So, server will get matrices of sizes (3+4)=7 from cluster1 and (5+6)=11 from cluster2 and 2 size from client5. Then, similarity is found as it is and client5 is assigned to new cluster based on similairty threshold. Then local phase(phase 4) happens then global phase(phase5) happens same as phase3. and all phases 7,9 are same as phase3. And phase 6,8,10 are local phases. But in phase 11, we will remove those cluster data and start as new so it will be same as phase1(so phase11 would be same as phase1) and everything above repeats after every RE_CLUSTER_INTERVAL = 10 phases.
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
MIN_CLUSTER_SIZE = 2         # Minimum clients per cluster
MIN_ROUNDS = 2               # Minimum expected rounds per client
NUM_CLIENTS = 9              # Number of clients
INITIAL_TRAINING_ROUNDS = 4  # Rounds before disconnecting from server
LOCAL_TRAINING_INTERVAL = 4  # Rounds of local training before reconnecting
CLUSTER_MERGE_INTERVAL = 3   # Interval for checking cluster merging
RE_CLUSTER_INTERVAL = 6     # Interval for re-clustering in phases

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
    """Create balanced client shards"""
    client_names = [f'client_{i+1}' for i in range(num_clients)]
    data, labels = shuffle(data, labels, random_state=42)
    
    shard_size = len(data) // num_clients
    shards = [(data[i*shard_size:(i+1)*shard_size], labels[i*shard_size:(i+1)*shard_size]) 
             for i in range(num_clients)]
    
    return {client_names[i]: shards[i] for i in range(num_clients)}

def prepare_data(df, scaler=None, fit=True):
    """Preprocess numerical data only"""
    # Clean data
    df = df.dropna()
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df = df[(df['fare_amount'] > 1) & (df['fare_amount'] < 5000)]
    
    # Feature processing
    X = df.drop('fare_amount', axis=1).select_dtypes(include=np.number)
    y = df['fare_amount']
    
    # Scale numerical features
    if fit or scaler is None:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if fit else scaler.transform(X)
    
    return X_scaled.astype(np.float32), y.values.astype(np.float32), scaler

def preprocess_data(train_path, test_path):
    """Consistent preprocessing for both train and test"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Process training data
    X_train, y_train, encoder, scaler = prepare_data(train_df, fit=True)
    
    # Process test data with same transformers
    X_test, y_test, _, _ = prepare_data(test_df, encoder=encoder, scaler=scaler, fit=False)
    
    return X_train, y_train, X_test, y_test

def load_client_data(client_path, scaler):
    """Load and preprocess individual client data"""
    df = pd.read_csv(client_path)
    X, y, _ = prepare_data(df, scaler=scaler, fit=False)
    return X, y

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
        initial_weights = self.base_model.get_weights()
        self.client_weights = {name: initial_weights for name in clients_data}
        self.round_wise_weights = {name: [] for name in clients_data}
        self.current_interval_weights = {name: [] for name in clients_data}
        self.client_weight_histories = {name: [] for name in clients_data}
        self.time_interval = INITIAL_TIME_INTERVAL
        self.history = []
        self.clusters = []
        self.final_clusters = None
        self.cluster_matrices = {}
        self.global_round_num = 0
        self.phase_counter = 1
        self.aggregation_mode = 'fedavg'
        

    def _create_compiled_model(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
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
        return [np.mean(layer, axis=0) for layer in zip(*client_weights_list)]


    def aggregate_weights(self, aggregation_algo='fedavg'):
        """Aggregation handler with algorithm selection"""
        weight_list = list(self.client_weights.values())
        
        if aggregation_algo == 'fedavg':
            return self.federated_average(weight_list)
        # Add other aggregation methods here
        else:
            raise ValueError(f"Unknown aggregation algorithm: {aggregation_algo}")

    # Modified train_round method in the FederatedTrainer class

    def train_round(self, X_test, y_test,phase_number):
        self.global_round_num += 1
        new_weights = {}
        round_counts = {}
        self.current_interval_weights = {client: [] for client in self.clients_data}

        # Local training phase
        for client, (X, y) in self.clients_data.items():
            try:
                model = self._clone_compiled_model()
                model.set_weights(self.client_weights[client])
                
                client_start = time.time()
                rounds = 0
                while time.time() - client_start < self.time_interval:
                    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                    self.current_interval_weights[client].append(model.get_weights())
                    rounds += 1
                
                new_weights[client] = model.get_weights()
                round_counts[client] = rounds
                self.round_wise_weights[client].append(new_weights[client])
                print(f"{client}: {rounds} rounds")
                
            except Exception as e:
                print(f"Error training {client}: {str(e)}")
                new_weights[client] = self.client_weights[client]

        if(phase_number != 1):
            imp_clients = self.select_clients_from_clusters()
            imp_weights = []
            for client in imp_clients:
                imp_weights.append(self.round_wise_weights[client][-1])
            aggregated_weights = self.federated_average(list(imp_weights))
            for client in self.clients_data:
                self.client_weights[client] = aggregated_weights
                self.client_weight_histories[client].append(aggregated_weights)
            self._record_history(round_counts, X_test, y_test)
            rmse, accuracy = self.evaluate(X_test, y_test)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
            return

        # Aggregate weights
        aggregated_weights = self.federated_average(list(new_weights.values()))
        for client in self.clients_data:
            self.client_weights[client] = aggregated_weights
            self.client_weight_histories[client].append(aggregated_weights)

        # Final round clustering
        if self.global_round_num == INITIAL_TRAINING_ROUNDS:
            self._perform_final_clustering()

        self._record_history(round_counts, X_test, y_test)
        rmse, accuracy = self.evaluate(X_test, y_test)
        print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")


    def select_clients_from_clusters(self):
        """Randomly select one client per cluster (for phase 3)"""
        selected = []
        for cluster in self.final_clusters:
            selected.append(np.random.choice(cluster))
        return selected

    def cluster_based_aggregation(self):
        """Aggregate weights from selected cluster representatives"""
        if not self.final_clusters:
            print("No clusters available. Using FedAvg instead.")
            return self.federated_average(list(self.client_weights.values()))
        
        selected = []
        for cluster in self.final_clusters:
            if cluster:  # Skip empty clusters
                selected.append(np.random.choice(cluster))
        
        if not selected:
            print("No clients selected from clusters. Using FedAvg.")
            return self.federated_average(list(self.client_weights.values()))
    
        return self.federated_average([self.client_weights[c] for c in selected])

    def add_new_client(self, client_name):
        """Add new client with data from reserved pool"""
        if self.full_dataset is None:
            raise ValueError("Full dataset not available for new clients")
        
        # Create new client data (same size as others)
        X, y = self.full_dataset
        client_data = create_clients(X, y, num_clients=1)
        new_client_data = list(client_data.values())[0]
        
        # Initialize with current global weights
        self.clients_data[client_name] = new_client_data
        self.client_weights[client_name] = self.base_model.get_weights()
        self.round_wise_weights[client_name] = []
        self.client_weight_histories[client_name] = []

    def full_reset(self):
        """Reset only cluster information while preserving clients and their weights"""
        print("\n=== Cluster Reset (Maintaining Clients/Weights) ===")
        # Reset cluster-related attributes only
        self.final_clusters = None
        self.clusters = []
        self.cluster_matrices = {}
        
        # Reset training phase counters
        self.global_round_num = 0
        self.phase_counter = 0
        
        print("Preserved:")
        print(f"- Clients: {list(self.clients_data.keys())}")
        print(f"- Client weight histories: {[len(h) for h in self.client_weight_histories.values()]}")

    def _get_cluster_matrix(self, cluster, round_num=None):
        """Get matrix for specific round or all rounds"""
        cluster_weights = []
        for client in cluster:
            if round_num is not None:
                weights = self.round_wise_weights[client][round_num]
            else:
                weights = [w for r in self.round_wise_weights[client] for w in r]
            cluster_weights.append(np.concatenate([w.flatten() for w in weights]))
        return np.column_stack(cluster_weights) if cluster_weights else np.array([])
    
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
        """Perform initial clustering after phase 1"""
        print("\n=== Initial Clustering ===")
        similarity_matrix = np.zeros((NUM_CLIENTS, NUM_CLIENTS))
        clients = list(self.clients_data.keys())
        
        for i, c1 in enumerate(clients):
            for j, c2 in enumerate(clients[i+1:], i+1):
                similarity = sum(
                    cosine_weight_similarity(w1, w2)
                    for w1 in self.current_interval_weights[c1]
                    for w2 in self.current_interval_weights[c2]
                )
                similarity_matrix[i,j] = similarity_matrix[j,i] = similarity
                print(f"client {i} v/s client {j} : {similarity}")
        
        # Form clusters based on similarity threshold
        self.clusters = []
        clustered = set()
        for i in range(NUM_CLIENTS):
            if i not in clustered:
                cluster = [clients[i]]
                for j in range(i+1, NUM_CLIENTS):
                    if similarity_matrix[i,j] >= SIMILARITY_THRESHOLD * MIN_ROUNDS:
                        cluster.append(clients[j])
                        clustered.add(j)
                self.clusters.append(cluster)
                clustered.add(i)
        print(self.final_clusters)
        self.final_clusters = self.clusters

    def _record_history(self, round_counts, X_test, y_test):
        """Modified to remove clustering from history recording"""
        diffs = {}
        base_weights = self.base_model.get_weights()
        
        if not base_weights:
            print("Base model weights are empty. Skipping history recording.")
            return
        
        flat_base = np.concatenate([w.flatten() for w in base_weights])
        
        for client in self.client_weights:
            client_weights = self.client_weights[client]
            if not client_weights:
                print(f"Skipping {client} due to empty weights.")
                diffs[client] = 0.0
                continue
            
            try:
                flat_client = np.concatenate([w.flatten() for w in client_weights])
                diffs[client] = np.linalg.norm(flat_client - flat_base)
            except ValueError as e:
                print(f"Error processing {client} weights: {str(e)}")
                diffs[client] = 0.0

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
    DATA_DIR = '/mnt/E_shared/project/Taxi_fare/data'
    TEST_FILE = '/mnt/E_shared/project/test_dataset_2000.csv'
    
    # Load first client to fit scaler
    print("Fitting global scaler...")
    first_client = pd.read_csv(os.path.join(DATA_DIR, 'client_1.csv'))
    _, _, global_scaler = prepare_data(first_client, fit=True)
    
    # Load all clients
    clients_data = {}
    for i in range(1,10):
        client_file = os.path.join(DATA_DIR, f'client_{i}.csv')
        X, y = load_client_data(client_file, global_scaler)
        clients_data[f'client_{i}'] = (X, y)
    
    # Prepare test data
    test_df = pd.read_csv(TEST_FILE)
    X_test, y_test, _ = prepare_data(test_df, scaler=global_scaler, fit=False)
    
    # Initialize trainer
    trainer = FederatedTrainer(clients_data, input_dim=X_test.shape[1])
    
    # Training loop
    phase_number = 1
    while len(trainer.history) < 100:
        if phase_number % RE_CLUSTER_INTERVAL == 0:
            trainer.full_reset()
            phase_number = 1

        if phase_number == 1:
            print("\n=== Phase 1: Initial Global Phase ===")
            for round_num in range(INITIAL_TRAINING_ROUNDS):
                print(f"Global Round {round_num+1}/{INITIAL_TRAINING_ROUNDS}")
                trainer.train_round(X_test, y_test,phase_number)
            
            print("\nInitial Clusters:")
            for i, cluster in enumerate(trainer.final_clusters):
                print(f"Cluster {i+1}: {', '.join(cluster)}")

        elif phase_number % 2 == 0:
            print(f"\n=== Phase {phase_number}: Local Training ===")
            for _ in range(LOCAL_TRAINING_INTERVAL):
                trainer.train_locally()
                rmse, acc = trainer.evaluate(X_test, y_test)
                print(f"Local Update RMSE: {rmse:.4f}, Accuracy: {acc:.2f}%")

        else:
            print(f"\n=== Phase {phase_number}: Global Training ===")
            trainer.aggregation_mode = 'cluster_based' if phase_number > 1 else 'fedavg'
            
            for round_num in range(INITIAL_TRAINING_ROUNDS):
                if phase_number == 3 and round_num == 3:
                    print("Adding new client...")
                    # Add new client implementation here
                
                print(f"Global Round {round_num+1}/{INITIAL_TRAINING_ROUNDS}")
                trainer.train_round(X_test, y_test,phase_number)

        phase_number += 1

    # Visualization and reporting
    print("\nTraining Summary:")
    plot_training_progress(trainer.history)
    plot_cluster_evolution(trainer.history)
    plot_global_metrics(trainer.history)

