import pandas as pd
from sklearn.model_selection import train_test_split
import os

data_path = "/mnt/E_shared/project/Taxi_fare/data"

for i in range(1, 10):
    filename = f"client_{i}.csv"
    file_path = os.path.join(data_path, filename)
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Split into 80% train and 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save 80% back into the original file
    train_df.to_csv(file_path, index=False)
    
    # Save 20% into a new test file
    test_filename = f"client_{i}_test.csv"
    test_path = os.path.join(data_path, test_filename)
    test_df.to_csv(test_path, index=False)

print("Splitting complete. Files saved.")
