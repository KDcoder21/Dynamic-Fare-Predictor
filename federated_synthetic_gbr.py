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

# Configuration
INITIAL_TIME_INTERVAL = 25
GROUP_THRESHOLD = 100
MIN_CLUSTER_SIZE = 2 
NUM_CLIENTS = 20
INITIAL_TRAINING_ROUNDS = 10
LOCAL_TRAINING_INTERVAL = 10
CLUSTER_MERGE_INTERVAL = 30
RE_CLUSTER_INTERVAL = 10
INITIAL_ESTIMATORS = 100 
ADDITIONAL_ESTIMATORS = 50 
TREE_SIMILARITY_THRESHOLD = 0.7 

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
            return model_vector
            
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
        self.client_vectors = {}
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
        

    def _create_gb_model(self):
        return GradientBoostingRegressor(
            n_estimators=self.phase_estimators, 
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            warm_start=True
        )

    def _update_client_vectors(self):
        for name, model in self.client_models.items():
            self.client_vectors[name] = self.encoder.model_to_vector(model)
    
    def _update_phase_estimators(self):
        """Update estimator count based on current phase"""
        self.phase_estimators = INITIAL_ESTIMATORS + self.phase_counter * ADDITIONAL_ESTIMATORS
        print(f"\n=== Phase {self.phase_counter} Estimators: {self.phase_estimators} ===")

    def train_round(self, X_test, y_test):
        new_models = {}
        round_counts = {}
        for model in self.client_models.values():
            model.n_estimators = self.phase_estimators

        for client, (X_sparse, y) in self.clients_data.items():
            try:
                window = self.client_windows[client]
                X_window = X_sparse[window['start']:window['end']].toarray()
                y_window = y[window['start']:window['end']]
                
                model = self.client_models[client]
                
                # Initial training phase
                if not hasattr(model, 'estimators_') or model.n_estimators < INITIAL_ESTIMATORS:
                    model.n_estimators = INITIAL_ESTIMATORS
                    model.fit(X_window, y_window)
                # Incremental training phase
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
        self._adjust_interval(round_counts)
        self._record_history(round_counts, X_test, y_test)
        
        rmse, accuracy = self.evaluate(X_test, y_test)
        print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate(self, X_test, y_test, batch_size=1000):
        """Batch evaluation with feature dimension check"""
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

    def _form_initial_clusters(self, active_clients):
        client_vectors = self.client_vectors
        print("\n=== Initial Cluster Formation ===")
        similarity_matrix = np.zeros((len(active_clients), len(active_clients)))
        clients = list(active_clients)
        
        print("\nPairwise Similarity Scores:")
        for i, c1 in enumerate(clients):
            for j, c2 in enumerate(clients[i+1:], start=i+1):
                vec1 = self.client_vectors[c1].reshape(1, -1)
                vec2 = self.client_vectors[c2].reshape(1, -1)
                sim = cosine_similarity(vec1, vec2)[0][0]
                status = "PASS" if sim >= TREE_SIMILARITY_THRESHOLD else "FAIL"
                print(f"{c1} vs {c2}: {sim:.2f} ({status})")
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        clusters = []
        visited = set()
        for idx, client in enumerate(clients):
            if client not in visited:
                cluster = [client]
                visited.add(client)
                for jdx, other in enumerate(clients):
                    if similarity_matrix[idx, jdx] >= TREE_SIMILARITY_THRESHOLD:
                        cluster.append(other)
                        visited.add(other)
                if len(cluster) >= MIN_CLUSTER_SIZE:
                    clusters.append(cluster)
        return clusters

    def _finalize_clusters(self):
        print("\n=== Final Cluster Merging ===")
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

    def _record_history(self, round_counts, X_test, y_test):
        """Track training progress and prepare data for visualizations"""
        
        # 1. Store client model vectors (2D arrays)
        client_vectors = {client: vec.copy() for client, vec in self.client_vectors.items()}
        avg_vector = np.mean(list(client_vectors.values()), axis=0)
        
        # 2. Cluster formation/update
        active_clients = [c for c in self.clients_data if round_counts.get(c, 0) > 0]
        if not self.final_clusters:
            self.clusters = self._form_initial_clusters(active_clients)
        else:
            self._handle_new_clients()

        # 3. Global evaluation
        global_rmse, global_accuracy = self.evaluate(X_test, y_test)
        
        # 4. Cluster-specific metrics
        cluster_metrics = {}
        cluster_features = {}
        feature_importances = {}
        
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_models = [self.client_models[c] for c in cluster]
            
            # Cluster predictions
            cluster_preds = np.mean([model.predict(X_test) for model in cluster_models], axis=0)
            cluster_metrics[cluster_idx] = {
                'rmse': np.sqrt(mean_squared_error(y_test, cluster_preds)),
                'accuracy': calculate_accuracy(y_test, cluster_preds),
                'size': len(cluster)
            }
            
            # Feature importance tracking
            if hasattr(cluster_models[0], 'feature_importances_'):
                avg_features = np.mean([model.feature_importances_ for model in cluster_models], axis=0)
                cluster_features[cluster_idx] = avg_features
                
                if cluster_idx not in feature_importances:
                    feature_importances[cluster_idx] = []
                feature_importances[cluster_idx].append(avg_features)
        
        # 5. Store complete history entry
        history_entry = {
            'round': len(self.history),
            'clusters': self.clusters.copy(),
            'client_vectors': client_vectors,
            'avg_vector': avg_vector,
            'interval': self.time_interval,
            'global_rmse': global_rmse,
            'global_accuracy': global_accuracy,
            'cluster_metrics': cluster_metrics,
            'cluster_features': cluster_features,
            'feature_importances': feature_importances,
            'active_clients': active_clients
        }
        
        self.history.append(history_entry)
        self.round_counter += 1
        
        # 6. Print summary
        print(f"\n=== Interval {len(self.history)} Report ===")
        print(f"Tree Similarity Threshold: {TREE_SIMILARITY_THRESHOLD}")
        print("\nFormed Clusters:")
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i+1}: {', '.join(cluster)}")
        print(f"Global RMSE: {global_rmse:.4f}, Accuracy: {global_accuracy:.2f}%")
        print("="*50)
        
        # 7. Generate visualizations at re-clustering phases
        # if self.round_counter % RE_CLUSTER_INTERVAL == 0:
        #     self._generate_visualizations()

    def _generate_visualizations(self):
        """Generate all monitoring visualizations"""
        print("\n=== Generating Cluster Monitoring Visualizations ===")
        
        # 1. Basic training progress
        plot_training_progress(self.history)
        
        # 2. Cluster evolution
        plot_cluster_evolution(self.history)
        
        # 3. Global metrics
        plot_global_metrics(self.history, initial_training_rounds=INITIAL_TRAINING_ROUNDS)
        
        # 4. Cluster size distribution
        plot_cluster_sizes(self.history)
        
        # 5. Cluster similarity analysis
        plot_cluster_similarity(self.history)
        
        # 6. Cluster performance comparison
        plot_cluster_performance(self.history)
        
        # 7. Client contribution heatmap
        plot_client_contributions(self.history)
        
        # 8. Feature importance evolution (if features available)
        if len(self.history) > 0 and 'cluster_features' in self.history[0]:
            try:
                feature_names = getattr(self, 'feature_names', 
                                    [f'Feature {i}' for i in range(self.n_features)])
                plot_feature_importance(self.history, feature_names)
            except Exception as e:
                print(f"Could not generate feature importance plot: {str(e)}")

    def _handle_new_clients(self):
        all_clients = set(self.clients_data.keys())
        clustered_clients = set().union(*self.final_clusters)
        new_clients = all_clients - clustered_clients

        for new_client in new_clients:
            print(f"\nEvaluating new client: {new_client}")
            max_similarity = -1
            best_cluster = None
            
            for cluster in self.final_clusters:
                cluster_vecs = [self.client_vectors[c] for c in cluster]
                client_vec = self.client_vectors[new_client]
                similarity = np.mean([cosine_similarity([cv], [client_vec])[0][0] for cv in cluster_vecs])
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = cluster
            
            if max_similarity >= TREE_SIMILARITY_THRESHOLD:
                print(f"Adding {new_client} to cluster {self.final_clusters.index(best_cluster)+1}")
                best_cluster.append(new_client)
            else:
                print(f"New cluster created for {new_client}")
                self.final_clusters.append([new_client])

    def _calculate_cluster_similarity(self, cluster1, cluster2):
        vecs1 = [self.client_vectors[c] for c in cluster1]
        vecs2 = [self.client_vectors[c] for c in cluster2]
        return np.mean([cosine_similarity([v1], [v2])[0][0] for v1 in vecs1 for v2 in vecs2])

    def train_locally(self):
        print("\n=== Local Training Phase ===")
        self._update_phase_estimators()
        for cluster in self.final_clusters:
            print(f"Training Cluster: {', '.join(cluster)}")
            cluster_models = [self.client_models[c] for c in cluster]
            
            for client in cluster:
                X, y = self.clients_data[client]
                model = self.client_models[client]
                model.n_estimators = self.phase_estimators
                pseudo_labels = np.mean([m.predict(X) for m in cluster_models], axis=0)
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

    def re_cluster(self):
        print("\n=== Re-clustering Process ===")
        self.clusters = []
        self.final_clusters = None
        
        active_clients = list(self.clients_data.keys())
        self.clusters = self._form_initial_clusters(active_clients)
        self._finalize_clusters()
        
        print("Re-clustering completed. New clusters:")
        for i, cluster in enumerate(self.final_clusters):
            print(f"Cluster {i+1}: {', '.join(cluster)}")

def create_clients(data, labels, num_clients=NUM_CLIENTS):
    """Create clients with sparse data chunks"""
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
    """Calculate accuracy as percentage of predictions within tolerance range"""
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance)
    return correct / len(y_true) * 100.0



def load_city_data(city_dir):
    """Load preprocessed data for a specific city"""
    return pd.read_csv(city_dir)

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
    
    
def prepare_data(df, encoder=None, chunk_size=1000):
    """Prepare data with consistent feature space"""
    # Initialize encoder if not provided
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
    
    # Initial centralized training phase
    print("\n=== Phase 1: Initial Centralized Phase ===")
    trainer.phase_counter = 0  # Start at phase 0
    trainer._update_phase_estimators()
    for round_num in range(INITIAL_TRAINING_ROUNDS):
        print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
        trainer.train_round(X_test, y_test)
        current_samples = {client: trainer.client_windows[client]['end'] 
                         for client in trainer.clients_data}
        print(f"Current samples per client: {current_samples}")
    
    trainer._finalize_clusters()
    trainer.phase_counter += 1

    # Main training loop
    while True:
        # Local training phase
        trainer.phase_counter += 1
        trainer._update_phase_estimators()
        print(f"\n=== Phase {trainer.phase_counter+1}: Local Training ===")
        for local_round in range(LOCAL_TRAINING_INTERVAL):
            print(f"\n=== Local Round {local_round+1}/{LOCAL_TRAINING_INTERVAL} ===")
            trainer.train_locally()
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
        
        trainer.phase_counter += 1
        trainer._update_phase_estimators()
        print(f"\n=== Phase {trainer.phase_counter}: Centralized Training ===")
        
        # Re-cluster and expand data every RE_CLUSTER_INTERVAL phases
        if trainer.phase_counter % RE_CLUSTER_INTERVAL == 0:
            print("\n=== Data Expansion and Re-clustering ===")
            trainer.re_cluster()
            
            # Visualize progress at each reclustering phase
            print("\n=== Cluster Monitoring Visualizations ===")
            plot_training_progress(trainer.history)
            plot_cluster_evolution(trainer.history)
            plot_global_metrics(trainer.history, initial_training_rounds=INITIAL_TRAINING_ROUNDS)
            plot_cluster_sizes(trainer.history)
            plot_cluster_similarity(trainer.history)
            plot_cluster_performance(trainer.history)
            plot_client_contributions(trainer.history)
            # plot_feature_importance(trainer.history,)
            
            # Expand data windows and increase model capacity
            for client in trainer.clients_data:
                window = trainer.client_windows[client]
                new_end = min(window['end'] + 500, len(trainer.clients_data[client][1]))
                trainer.client_windows[client]['end'] = new_end
                #trainer.client_models[client].n_estimators += 50
                
                print(f"{client}: Expanded to {new_end} samples | "
                      f"Now {trainer.client_models[client].n_estimators} trees")

        # Centralized training phase
        print(f"\n=== Phase {trainer.phase_counter+1}: Centralized Training ===")
        for round_num in range(INITIAL_TRAINING_ROUNDS):
            print(f"\n=== Round {round_num+1}/{INITIAL_TRAINING_ROUNDS} ===")
            trainer.train_round(X_test, y_test)
            rmse, accuracy = trainer.evaluate(X_test, y_test)
            global_rmse.append(rmse)
            global_accuracy.append(accuracy)
            print(f"Global RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")
        
        trainer.phase_counter += 1
        
        # Termination condition
        if len(global_rmse) >= 100:
            print("\nReached maximum training rounds (100)")
            break

    # Final results
    print("\nTraining Summary:")
    print(f"Best RMSE: {min(global_rmse):.4f}")
    print(f"Best Accuracy: {max(global_accuracy):.2f}%")
    
    print("\nFinal Client Statistics:")
    for client in trainer.clients_data:
        window = trainer.client_windows[client]
        active_rounds = sum(1 for entry in trainer.history 
                           if entry['diffs'][client] > 0)
        print(f"{client}:")
        print(f"  - Final samples used: {window['end']}/{len(trainer.clients_data[client][1])}")
        print(f"  - Final trees: {trainer.client_models[client].n_estimators}")
        print(f"  - Participation: {active_rounds}/{len(trainer.history)} rounds")

    # Visualization
    plot_training_progress(trainer.history)
    plot_cluster_evolution(trainer.history)
    plot_global_metrics(trainer.history, initial_training_rounds=INITIAL_TRAINING_ROUNDS)