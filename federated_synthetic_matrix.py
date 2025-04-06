import numpy as np
import pandas as pd
import time
import os
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import _tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from scipy.sparse import hstack, vstack, csr_matrix
from collections import defaultdict

# Configuration
INITIAL_TIME_INTERVAL = 25
SIMILARITY_THRESHOLD = 0.7
GROUP_THRESHOLD = 100
MIN_CLUSTER_SIZE = 2
NUM_CLIENTS = 2
INITIAL_TRAINING_ROUNDS = 2
LOCAL_TRAINING_INTERVAL = 4
CLUSTER_MERGE_INTERVAL = 3
RE_CLUSTER_INTERVAL = 2
INITIAL_ESTIMATORS = 100 
ADDITIONAL_ESTIMATORS = 50 
TREE_SIMILARITY_THRESHOLD = 0.7
PAIR_SIMILARITY_THRESHOLD = 0.65  # Threshold for individual weight vector pairs
CLIENT_SIMILARITY_THRESHOLD = 3    # Minimum similar pairs to group clients

class TreeStructureEncoder:
    def __init__(self, n_features, max_depth=5):
        self.n_features = n_features
        self.max_depth = max_depth
        self.vector_size = n_features * (max_depth + 1)
        
    def _tree_to_vector(self, tree):
        vector = np.zeros(self.vector_size)
        stack = [(0, 0, tree)]
        
        while stack:
            node_idx, depth, t = stack.pop()
            
            if depth > self.max_depth:
                continue
                
            if t.children_left[node_idx] == _tree.TREE_LEAF:
                vector[self.n_features * self.max_depth:] += t.value[node_idx][0]
                continue
                
            feature = t.feature[node_idx]
            threshold = t.threshold[node_idx]
            
            start_idx = self.n_features * depth
            vector[start_idx + feature] += 1
            vector[start_idx + self.n_features//2 + feature] += threshold
            
            stack.append((t.children_left[node_idx], depth+1, t))
            stack.append((t.children_right[node_idx], depth+1, t))
            
        return vector
    
    def model_to_vector(self, gb_model):
        model_vector = np.zeros(self.vector_size)
        if not hasattr(gb_model, 'estimators_'):
            return model_vector  # Return zero vector if model not trained yet
            
        for tree in gb_model.estimators_:
            tree_vector = self._tree_to_vector(tree[0].tree_)
            model_vector += tree_vector
        return model_vector / len(gb_model.estimators_)

class FederatedGBMTrainer:
    def __init__(self, clients_data, n_features):
        self.clients_data = clients_data
        self.n_features = n_features
        self.encoder = TreeStructureEncoder(n_features)
        self.phase_estimators = INITIAL_ESTIMATORS 
        self.client_models = {name: self._create_gb_model() for name in clients_data}
        self.client_vectors = defaultdict(list)  # Stores all vectors per client per interval
        self.current_interval_vectors = {}  # Temporary storage for current interval
        self.time_interval = INITIAL_TIME_INTERVAL
        self.history = []
        self.clusters = []
        self.final_clusters = None
        self.round_counter = 0
        self.phase_counter = 0
        self.client_windows = {
            client: {'start': 0, 'end': 500} 
            for client in clients_data
        }
        self.interval_start_time = time.time()

    def _create_gb_model(self):
        return GradientBoostingRegressor(
            n_estimators=self.phase_estimators,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            warm_start=True
        )

    def _update_client_vectors(self):
        """Update vectors for current training round"""
        for name, model in self.client_models.items():
            current_vector = self.encoder.model_to_vector(model)
            self.current_interval_vectors[name] = current_vector

    def _check_interval_elapsed(self):
        """Check if fixed time interval has passed"""
        return time.time() - self.interval_start_time >= self.time_interval

    def _process_interval_vectors(self):
        """Store all vectors from current interval and reset"""
        for client, vector in self.current_interval_vectors.items():
            self.client_vectors[client].append(vector)
        self.current_interval_vectors = {}
        self.interval_start_time = time.time()

    def _update_phase_estimators(self):
        """Update estimator count when phase changes"""
        self.phase_estimators = INITIAL_ESTIMATORS + self.phase_counter * ADDITIONAL_ESTIMATORS
        print(f"\n=== Phase {self.phase_counter} Estimators: {self.phase_estimators} ===")

    def train_round(self, X_test, y_test):
        new_models = {}
        round_counts = {}
        
        for client, (X_sparse, y) in self.clients_data.items():
            try:
                window = self.client_windows[client]
                X_window = X_sparse[window['start']:window['end']].toarray()
                y_window = y[window['start']:window['end']]
                
                model = self.client_models[client]
                
                if not hasattr(model, 'estimators_') or model.n_estimators < INITIAL_ESTIMATORS:
                    model.n_estimators = INITIAL_ESTIMATORS
                    model.fit(X_window, y_window)
                else:
                    model.n_estimators += ADDITIONAL_ESTIMATORS
                    model.fit(X_window, y_window)
                
                new_models[client] = model
                round_counts[client] = model.n_estimators
                print(f"{client}: {model.n_estimators} trees | Using {window['end']} samples")
                
            except Exception as e:
                print(f"Error training {client}: {str(e)}")
                new_models[client] = self.client_models[client]
        
        self.client_models = new_models
        self._update_client_vectors()
        
        # Check if interval completed
        if self._check_interval_elapsed():
            self._process_interval_vectors()
            self._form_clusters_based_on_intervals()
        
        self._adjust_interval(round_counts)
        self._record_history(round_counts, X_test, y_test)
        
        rmse, accuracy = self.evaluate(X_test, y_test)
        print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

    def _form_clusters_based_on_intervals(self):
        """New clustering logic based on interval vectors"""
        print("\n=== Interval-based Cluster Formation ===")
        
        # Step 1: Calculate pairwise similarity counts
        similarity_counts = defaultdict(int)
        clients = list(self.client_vectors.keys())
        
        # Get all possible vector pairs between clients
        for c1, c2 in combinations(clients, 2):
            vecs1 = self.client_vectors[c1]
            vecs2 = self.client_vectors[c2]
            
            # Calculate similarity for all combinations of their vectors
            for v1 in vecs1:
                for v2 in vecs2:
                    sim = cosine_similarity([v1], [v2])[0][0]
                    if sim >= PAIR_SIMILARITY_THRESHOLD:
                        similarity_counts[(c1, c2)] += 1
        
        # Step 2: Form clusters based on similarity counts
        clusters = []
        visited = set()
        
        for client in clients:
            if client not in visited:
                cluster = [client]
                visited.add(client)
                
                # Find all clients with sufficient similar pairs
                for other in clients:
                    if other != client and other not in visited:
                        pair = tuple(sorted((client, other)))
                        if similarity_counts.get(pair, 0) >= CLIENT_SIMILARITY_THRESHOLD:
                            cluster.append(other)
                            visited.add(other)
                
                if len(cluster) >= MIN_CLUSTER_SIZE:
                    clusters.append(cluster)
        
        # Step 3: Merge highly similar clusters
        new_clusters = []
        merged = set()
        
        for i, cluster1 in enumerate(clusters):
            if tuple(cluster1) in merged:
                continue
                
            current_cluster = set(cluster1)
            for j, cluster2 in enumerate(clusters[i+1:]):
                # Calculate inter-cluster similarity
                inter_similar = 0
                for c1 in cluster1:
                    for c2 in cluster2:
                        pair = tuple(sorted((c1, c2)))
                        inter_similar += similarity_counts.get(pair, 0)
                
                avg_inter_similar = inter_similar / (len(cluster1) * len(cluster2))
                if avg_inter_similar >= CLIENT_SIMILARITY_THRESHOLD:
                    current_cluster.update(cluster2)
                    merged.add(tuple(cluster2))
            
            new_clusters.append(list(current_cluster))
            merged.add(tuple(cluster1))
        
        print(f"Formed {len(new_clusters)} clusters from {len(clusters)} initial groups")
        self.final_clusters = new_clusters
        self.clusters = new_clusters
        
        # Print cluster diagnostics
        print("\nCluster Assignments:")
        for i, cluster in enumerate(self.final_clusters):
            print(f"Cluster {i+1}: {', '.join(cluster)}")
        
        # Print similarity diagnostics
        print("\nTop Similar Client Pairs:")
        sorted_pairs = sorted(similarity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for (c1, c2), count in sorted_pairs:
            print(f"{c1} & {c2}: {count} similar vector pairs")

    def evaluate(self, X_test, y_test, batch_size=1000):
        preds = []
        for i in range(0, X_test.shape[0], batch_size):
            batch = X_test[i:i+batch_size].toarray()
            
            for model in self.client_models.values():
                if batch.shape[1] != model.n_features_in_:
                    raise ValueError(
                        f"Feature mismatch: Model expects {model.n_features_in_} features, "
                        f"got {batch.shape[1]}"
                    )
            
            batch_preds = np.mean([m.predict(batch) for m in self.client_models.values()], axis=0)
            preds.extend(batch_preds)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        accuracy = calculate_accuracy(y_test, preds)
        return rmse, accuracy

    def _record_history(self, round_counts, X_test, y_test):
        client_vectors = {
            client: (self.client_vectors[client][-1] if self.client_vectors[client] 
                    else np.zeros(self.encoder.vector_size))
            for client in self.clients_data
        }
        avg_vector = np.mean(list(client_vectors.values()), axis=0)
        
        active_clients = [c for c in self.clients_data if round_counts.get(c, 0) > 0]
        
        global_rmse, global_accuracy = self.evaluate(X_test, y_test)
        
        cluster_metrics = {}
        for cluster_idx, cluster in enumerate(self.final_clusters if self.final_clusters else []):
            cluster_models = [self.client_models[c] for c in cluster]
            cluster_preds = np.mean([model.predict(X_test) for model in cluster_models], axis=0)
            cluster_metrics[cluster_idx] = {
                'rmse': np.sqrt(mean_squared_error(y_test, cluster_preds)),
                'accuracy': calculate_accuracy(y_test, cluster_preds),
                'size': len(cluster)
            }
        
        history_entry = {
            'round': len(self.history),
            'clusters': self.final_clusters.copy() if self.final_clusters else [],
            'client_vectors': client_vectors,
            'avg_vector': avg_vector,
            'interval': self.time_interval,
            'global_rmse': global_rmse,
            'global_accuracy': global_accuracy,
            'cluster_metrics': cluster_metrics,
            'active_clients': active_clients
        }
        
        self.history.append(history_entry)
        self.round_counter += 1
        
        print(f"\n=== Interval {len(self.history)} Report ===")
        print(f"Global RMSE: {global_rmse:.4f}, Accuracy: {global_accuracy:.2f}%")
        print("="*50)

    def train_locally(self):
        print("\n=== Local Training Phase ===")
        self.phase_counter += 1
        self._update_phase_estimators()

        for cluster in self.final_clusters:
            print(f"Training Cluster: {', '.join(cluster)}")
            cluster_models = [self.client_models[c] for c in cluster]
            
            for client in cluster:
                X, y = self.clients_data[client]
                pseudo_labels = np.mean([m.predict(X) for m in cluster_models], axis=0)
                
                model = self.client_models[client]
                model.n_estimators = self.phase_estimators
                model.fit(X, pseudo_labels)

    def _adjust_interval(self, round_counts):
        valid_rounds = [r for r in round_counts.values() if r > 0]
        if not valid_rounds:
            return
            
        avg_rounds = np.mean(valid_rounds)
        if avg_rounds < INITIAL_ESTIMATORS:
            self.time_interval = min(self.time_interval * 1.2, 60)
        else:
            self.time_interval = max(self.time_interval * 0.9, 15)

# Helper functions (keep the same as before)
def create_clients(data, labels, num_clients=NUM_CLIENTS):
    client_names = [f'client_{i+1}' for i in range(num_clients)]
    clients = {}
    total_samples = data.shape[0]
    samples_per_client = total_samples // num_clients
    
    for i, name in enumerate(client_names):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        if i == num_clients - 1:
            end_idx = total_samples
        
        X_client = data[start_idx:end_idx]
        y_client = labels[start_idx:end_idx]
        clients[name] = (X_client, y_client)
    
    return clients

def calculate_accuracy(y_true, y_pred, tolerance=5.0):
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance)
    return correct / len(y_true) * 100.0

def load_city_data(city_dir):
    return pd.read_csv(city_dir)

def prepare_data(df, encoder=None, chunk_size=1000):
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount'].values
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        if not hasattr(encoder, 'feature_names_in_'):
            encoder.fit(X[categorical_cols])
        encoded_data = encoder.transform(X[categorical_cols])
        numeric_data = X.drop(categorical_cols, axis=1).values
        X_sparse = hstack([encoded_data, csr_matrix(numeric_data)])
    else:
        X_sparse = csr_matrix(X.values)
    
    return X_sparse, y, encoder

# Visualization functions
def plot_training_progress(history):
    plt.figure(figsize=(15, 5))
    
    # Active clients plot
    plt.subplot(1, 3, 1)
    active_counts = [len(entry['active_clients']) for entry in history]
    plt.plot(active_counts, marker='o')
    plt.title('Active Clients per Round')
    plt.xlabel('Round')
    plt.grid(True)
    
    # Interval adjustment plot
    plt.subplot(1, 3, 2)
    intervals = [entry['interval'] for entry in history]
    plt.plot(intervals, marker='s', color='orange')
    plt.title('Training Interval Adjustment')
    plt.xlabel('Round')
    plt.grid(True)
    
    # Model differences plot
    plt.subplot(1, 3, 3)
    for client in history[0]['client_vectors']:
        diffs = []
        for entry in history:
            if client in entry['client_vectors']:
                # Calculate difference from average vector
                diff = np.linalg.norm(
                    entry['client_vectors'][client] - entry['avg_vector']
                )
                diffs.append(diff)
        plt.plot(diffs, alpha=0.5, label=client)
    
    plt.title('Model Structure Differences')
    plt.xlabel('Round')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cluster_evolution(history):
    """Visualize cluster membership changes over time"""
    if not history:
        return
    
    all_clients = list(history[0]['client_vectors'].keys())
    num_rounds = len(history)
    cluster_matrix = np.zeros((len(all_clients), num_rounds))
    
    for round_idx, entry in enumerate(history):
        for cluster_idx, cluster in enumerate(entry['clusters']):
            for client in cluster:
                client_idx = all_clients.index(client)
                cluster_matrix[client_idx, round_idx] = cluster_idx + 1

    plt.figure(figsize=(12, 6))
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

def plot_cluster_sizes(history):
    """Plot cluster size distribution over time"""
    plt.figure(figsize=(10, 5))
    for entry in history:
        cluster_sizes = [len(cluster) for cluster in entry['clusters']]
        plt.scatter([entry['round']]*len(cluster_sizes), cluster_sizes, c='blue', alpha=0.5)
    
    plt.title('Cluster Size Distribution Over Time')
    plt.xlabel('Communication Round')
    plt.ylabel('Cluster Size')
    plt.grid(True)
    plt.show()

def plot_cluster_similarity(history):
    """Plot intra-cluster and inter-cluster similarity with diagnostics"""
    intra_sims = []
    inter_sims = []
    
    print("\n=== Similarity Diagnostics ===")
    
    for idx, entry in enumerate(history):
        clusters = entry['clusters']
        client_vectors = entry['client_vectors']
        
        print(f"\nRound {idx} Cluster Structure: {[len(c) for c in clusters]}")
        
        # Intra-cluster similarity
        intra = []
        for c_idx, cluster in enumerate(clusters):
            if len(cluster) < 2:
                print(f"  Cluster {c_idx}: Insufficient clients ({len(cluster)})")
                continue
                
            vectors = [client_vectors[c].reshape(1, -1) for c in cluster]
            try:
                similarities = cosine_similarity(np.vstack(vectors))
                cluster_avg = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                intra.append(cluster_avg)
                print(f"  Cluster {c_idx} Intra-sim: {cluster_avg:.2f}")
            except Exception as e:
                print(f"  Cluster {c_idx} Error: {str(e)}")
                continue
        
        # Inter-cluster similarity
        inter = []
        for (c1, c2) in combinations(clusters, 2):
            vecs1 = [client_vectors[client].reshape(1, -1) for client in c1]
            vecs2 = [client_vectors[client].reshape(1, -1) for client in c2]
            
            try:
                similarities = cosine_similarity(np.vstack(vecs1), np.vstack(vecs2))
                pair_avg = np.mean(similarities)
                inter.append(pair_avg)
                print(f"  Cluster {c1}-{c2} Inter-sim: {pair_avg:.2f}")
            except Exception as e:
                print(f"  Inter-cluster Error: {str(e)}")
                continue
        
        intra_sims.append(np.mean(intra) if intra else np.nan)
        inter_sims.append(np.mean(inter) if inter else np.nan)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    valid_mask = ~np.isnan(intra_sims)
    if any(valid_mask):
        plt.plot(np.array(intra_sims)[valid_mask], label='Intra-cluster', marker='o')
    
    valid_mask = ~np.isnan(inter_sims)
    if any(valid_mask):
        plt.plot(np.array(inter_sims)[valid_mask], label='Inter-cluster', marker='s')
    
    plt.title('Cluster Similarity Evolution')
    plt.xlabel('Communication Round')
    plt.ylabel('Cosine Similarity')
    plt.axhline(TREE_SIMILARITY_THRESHOLD, color='r', linestyle='--', 
               label=f'Threshold ({TREE_SIMILARITY_THRESHOLD})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(intra_sims, label='Intra-cluster Similarity')
    plt.plot(inter_sims, label='Inter-cluster Similarity')
    plt.title('Cluster Similarity Evolution')
    plt.xlabel('Communication Round')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cluster_performance(history):
    """Compare RMSE across different clusters"""
    plt.figure(figsize=(10, 5))
    for idx, entry in enumerate(history):
        if 'cluster_metrics' in entry:
            for cluster_id, metrics in entry['cluster_metrics'].items():
                plt.scatter(idx, metrics['rmse'], 
                          c=f'C{cluster_id}', label=f'Cluster {cluster_id}' if idx == 0 else "")
    
    plt.title('Cluster-wise RMSE Comparison')
    plt.xlabel('Communication Round')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_client_contributions(history):
    """Visualize client participation in clusters"""
    if not history:
        return
    
    all_clients = list(history[0]['client_vectors'].keys())
    num_rounds = len(history)
    contrib_matrix = np.zeros((len(all_clients), num_rounds))
    
    for round_idx, entry in enumerate(history):
        client_assignments = {}
        for cluster_idx, cluster in enumerate(entry['clusters']):
            for client in cluster:
                client_assignments[client] = cluster_idx
                
        for client in all_clients:
            if client in client_assignments:
                client_idx = all_clients.index(client)
                contrib_matrix[client_idx, round_idx] = client_assignments[client] + 1

    plt.figure(figsize=(12, 6))
    plt.imshow(contrib_matrix, aspect='auto', cmap='tab20')
    plt.colorbar(label='Cluster ID')
    plt.yticks(range(len(all_clients)), all_clients)
    plt.xlabel("Communication Round")
    plt.ylabel("Client")
    plt.title("Client Cluster Participation Over Time")
    plt.show()

def plot_feature_importance(history, feature_names):
    """Track feature importance evolution in clusters"""
    plt.figure(figsize=(12, 6))
    for entry in history:
        if 'cluster_features' in entry:
            for cluster_id, features in entry['cluster_features'].items():
                plt.plot(features, alpha=0.5, label=f'Cluster {cluster_id}' if entry['round'] == 0 else "")
    
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.title('Feature Importance Evolution Across Clusters')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    train_file = '/mnt/E_shared/project/test_dataset.csv'
    test_file = '/mnt/E_shared/project/train_dataset.csv'
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    
    print("Loading and preprocessing training data...")
    train_df = load_city_data(train_file)
    X_train, y_train, encoder = prepare_data(train_df, encoder=encoder)
    
    print("Loading and preprocessing test data...")
    test_df = load_city_data(test_file)
    X_test, y_test, _ = prepare_data(test_df, encoder=encoder)

    assert X_train.shape[1] == X_test.shape[1], \
        f"Feature mismatch: train has {X_train.shape[1]} features, test has {X_test.shape[1]}"
    
    print(f"\nCreating {NUM_CLIENTS} clients with initial 500 samples each...")
    clients = create_clients(X_train, y_train)
    n_features = X_train.shape[1]
    
    trainer = FederatedGBMTrainer(clients, n_features)
    global_rmse = []
    global_accuracy = []
    
    print("\n=== Phase 1: Initial Centralized Phase ===")
    trainer._update_phase_estimators()
    for round_num in range(INITIAL_TRAINING_ROUNDS):
        print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
        trainer.train_round(X_test, y_test)
        current_samples = {client: trainer.client_windows[client]['end'] 
                         for client in trainer.clients_data}
        print(f"Current samples per client: {current_samples}")
    
    trainer.phase_counter = 1

    # Main training loop
    while True:
        print(f"\n=== Phase {trainer.phase_counter+1}: Local Training ===")
        trainer._update_phase_estimators()
        for local_round in range(LOCAL_TRAINING_INTERVAL):
            print(f"\n=== Local Round {local_round+1}/{LOCAL_TRAINING_INTERVAL} ===")
            trainer._update_phase_estimators()
            trainer.train_locally()
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
        
        trainer.phase_counter += 1
        
        if trainer.phase_counter % RE_CLUSTER_INTERVAL == 0:
            print("\n=== Data Expansion ===")
            for client in trainer.clients_data:
                window = trainer.client_windows[client]
                new_end = min(window['end'] + 500, len(trainer.clients_data[client][1]))
                trainer.client_windows[client]['end'] = new_end
                trainer.client_models[client].n_estimators += 50
                print(f"{client}: Expanded to {new_end} samples | Now {trainer.client_models[client].n_estimators} trees")

        print(f"\n=== Phase {trainer.phase_counter+1}: Centralized Training ===")
        trainer._update_phase_estimators()
        for round_num in range(INITIAL_TRAINING_ROUNDS):
            print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
            trainer.train_round(X_test, y_test)
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
        
        trainer.phase_counter += 1
        
        if len(global_rmse) >= 100:
            print("\nReached maximum training rounds (100)")
            break

    # Final results and visualizations
    print("\nTraining Summary:")
    print(f"Best RMSE: {min(global_rmse):.4f}")
    print(f"Best Accuracy: {max(global_accuracy):.2f}%")
    
    print("\nFinal Client Statistics:")
    for client in trainer.clients_data:
        window = trainer.client_windows[client]
        print(f"{client}:")
        print(f"  - Final samples used: {window['end']}/{len(trainer.clients_data[client][1])}")
        print(f"  - Final trees: {trainer.client_models[client].n_estimators}")

    plot_training_progress(trainer.history)
    plot_cluster_evolution(trainer.history)
    plot_global_metrics(trainer.history, initial_training_rounds=INITIAL_TRAINING_ROUNDS)