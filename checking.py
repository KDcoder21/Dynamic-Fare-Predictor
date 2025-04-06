import pandas as pd

# Path to your Parquet file
parquet_file = "data/green_tripdata_2022-12.parquet"

# Read the Parquet file
df = pd.read_parquet(parquet_file, engine='pyarrow')  # or engine='fastparquet'

# Display the first few rows of the DataFrame
print(df.head())
print(df.describe())  # Summary statistics for numeric columns
print(df.columns)  # List of column names