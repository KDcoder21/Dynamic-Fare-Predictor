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
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate, Lambda
from tensorflow.keras import backend as K  # Import Keras backend


# Configure environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuration parameters
INITIAL_TIME_INTERVAL = 25
SIMILARITY_THRESHOLD = 0.5
GROUP_THRESHOLD = 500
MIN_CLUSTER_SIZE = 2
MIN_ROUNDS = 2
NUM_CLIENTS = 10

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

def preprocess_data(file_path):
    """Robust data preprocessing pipeline"""
    df = pd.read_parquet(file_path)
    
    # Data cleaning and validation
    df = df.dropna(subset=[
        'fare_amount', 'PULocationID', 'DOLocationID',
        'trip_distance', 'passenger_count', 'lpep_pickup_datetime'
    ])
    df = df[(df['fare_amount'] > 1) & (df['fare_amount'] < 500)]
    df = df[df['passenger_count'].between(1, 8)]
    
    # Feature engineering
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['lpep_pickup_datetime'].dt.dayofweek
    
    # Split before processing locations
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create location mappings from training data only
    pu_map = {id: idx+1 for idx, id in enumerate(train_df['PULocationID'].unique())}
    do_map = {id: idx+1 for idx, id in enumerate(train_df['DOLocationID'].unique())}
    
    def remap_locations(df, pu_map, do_map):
        """Remap location IDs to continuous indices"""
        df['PULocationID'] = df['PULocationID'].map(pu_map).fillna(0).astype(int)
        df['DOLocationID'] = df['DOLocationID'].map(do_map).fillna(0).astype(int)
        return df
    
    train_df = remap_locations(train_df, pu_map, do_map)
    test_df = remap_locations(test_df, pu_map, do_map)
    
    # Prepare features
    numerical_features = ['trip_distance', 'passenger_count', 'pickup_hour', 'pickup_day']
    scaler = StandardScaler()
    
    X_train = np.column_stack([
        train_df['PULocationID'].values,
        train_df['DOLocationID'].values,
        scaler.fit_transform(train_df[numerical_features])
    ])
    
    X_test = np.column_stack([
        test_df['PULocationID'].values,
        test_df['DOLocationID'].values,
        scaler.transform(test_df[numerical_features])
    ])
    
    return (
        X_train, train_df['fare_amount'].values,
        X_test, test_df['fare_amount'].values,
        len(pu_map)+1, len(do_map)+1
    )

class FederatedTrainer:
    def __init__(self, clients_data, num_pu_locations, num_do_locations):
        self.clients_data = clients_data
        self.num_pu_locations = num_pu_locations
        self.num_do_locations = num_do_locations
        self.base_model = self._create_compiled_model()
        self.client_weights = {name: self.base_model.get_weights() for name in clients_data}
        self.current_interval_weights = {name: [] for name in clients_data}
        self.time_interval = INITIAL_TIME_INTERVAL
        self.history = []
        
    def _create_compiled_model(self):
        """Create and compile base model template"""
        input_layer = Input(shape=(6,))
        
        # Feature extraction using Lambda layers
        pu_id = Lambda(
            lambda x: x[:, 0], 
            output_shape=(1,)
        )(input_layer)
        do_id = Lambda(
            lambda x: x[:, 1], 
            output_shape=(1,)
        )(input_layer)
        numerical = Lambda(
            lambda x: x[:, 2:6], 
            output_shape=(4,)
        )(input_layer)
        
        # Cast to integer using tf directly
        pu_id = Lambda(
            lambda x: tf.cast(x, tf.int32),  # Use tf directly
            output_shape=(1,)
        )(pu_id)
        do_id = Lambda(
            lambda x: tf.cast(x, tf.int32),  # Use tf directly
            output_shape=(1,)
        )(do_id)
        
        # Embedding layers
        pu_embed = Embedding(
            input_dim=self.num_pu_locations+1, 
            output_dim=8,
            input_length=1
        )(pu_id)
        do_embed = Embedding(
            input_dim=self.num_do_locations+1,
            output_dim=8,
            input_length=1
        )(do_id)
        
        # Feature processing
        pu_flat = Flatten()(pu_embed)
        do_flat = Flatten()(do_embed)
        
        concat = Concatenate()([pu_flat, do_flat, numerical])
        x = Dense(64, activation='relu')(concat)
        x = Dense(32, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _clone_compiled_model(self):
        """Create compiled clone of base model"""
        model = clone_model(self.base_model)
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def train_round(self):
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
                    batch_size = min(32, max(16, len(X)//2))
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
        self._record_history(round_counts)
        
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
            
    def _record_history(self, round_counts):
        """Track training metrics and perform clustering"""
        diffs = {}
        base_weights = self.base_model.get_weights()
        flat_base = np.concatenate([w.flatten() for w in base_weights])
        
        for client in self.client_weights:
            flat_client = np.concatenate([w.flatten() for w in self.client_weights[client]])
            diffs[client] = np.linalg.norm(flat_client - flat_base)

        active_clients = [c for c in self.clients_data if round_counts.get(c, 0) > 0]
        similarity_counts = {}
        cluster_graph = {c: set() for c in active_clients}

        # Calculate pairwise similarities
        for (c1, c2) in combinations(active_clients, 2):
            count = 0
            for w1 in self.current_interval_weights[c1]:
                for w2 in self.current_interval_weights[c2]:
                    flat_w1 = np.concatenate([w.flatten() for w in w1])
                    flat_w2 = np.concatenate([w.flatten() for w in w2])
                    similarity = cosine_similarity([flat_w1], [flat_w2])[0][0]
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

        # Print similarity report
        print(f"\n=== Interval {len(self.history)+1} Report ===")
        print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
        print(f"Grouping Threshold: {GROUP_THRESHOLD} matching pairs")
        
        for pair, count in similarity_counts.items():
            status = "PASS" if count >= GROUP_THRESHOLD else "FAIL"
            print(f"{pair[0]} vs {pair[1]}: {count} pairs ({status})")
        
        print("\nFormed Clusters:")
        if clusters:
            for i, cluster in enumerate(clusters):
                print(f"Cluster {i+1}: {', '.join(cluster)}")
        else:
            print("No clusters formed")
        print("="*50)

        self.history.append({
            'round': len(self.history),
            'clusters': clusters,
            'diffs': diffs,
            'interval': self.time_interval,
            'similarities': similarity_counts
        })
        
    def aggregate_weights(self):
        """Federated averaging with fallback"""
        if not self.client_weights:
            return self.base_model.get_weights()
            
        return [np.mean(layer, axis=0) 
               for layer in zip(*self.client_weights.values())]
        
    def evaluate(self, X_test, y_test):
        """Robust model evaluation"""
        try:
            self.base_model.set_weights(self.aggregate_weights())
            preds = self.base_model.predict(X_test, verbose=0).flatten()
            return np.sqrt(mean_squared_error(y_test, preds))
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return float('inf')

# Visualization functions remain unchanged
def plot_training_progress(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    active_counts = [len(c for c in entry['diffs'] if entry['diffs'][c] > 0) for entry in history]
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

if __name__ == "__main__":
    # Load and preprocess data
    train_file = 'data/green_tripdata_2022-12.parquet'
    X_train, y_train, X_test, y_test, num_pu, num_do = preprocess_data(train_file)
    
    # Create federated clients
    clients = create_clients(X_train, y_train)
    trainer = FederatedTrainer(clients, num_pu, num_do)
    global_rmse = []
    
    # Training loop
    num_rounds = 10
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num+1}/{num_rounds} ===")
        print(f"Current interval: {trainer.time_interval:.1f}s")
        
        trainer.train_round()
        rmse = trainer.evaluate(X_test, y_test)
        global_rmse.append(rmse)
        print(f"Global RMSE: {rmse:.4f}")
    
    # Results analysis
    print("\nTraining Summary:")
    print(f"Best RMSE: {min(global_rmse):.4f}")
    print(f"Final Interval: {trainer.time_interval:.1f}s")
    print("Client Participation Summary:")
    for client in trainer.clients_data:
        active_rounds = sum(1 for entry in trainer.history 
                           if entry['diffs'][client] > 0)
        print(f"{client}: {active_rounds}/{num_rounds} rounds")

    plot_training_progress(trainer.history)
    plot_cluster_evolution(trainer.history)