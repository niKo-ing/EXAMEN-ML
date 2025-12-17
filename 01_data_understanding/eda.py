import pandas as pd
import os

def analyze_application_data():
    file_path = 'application_.parquet'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    print(f"\nDataset Shape: {df.shape}")
    
    print("\nTarget Distribution:")
    if 'TARGET' in df.columns:
        print(df['TARGET'].value_counts(normalize=True))
    else:
        print("TARGET column not found!")

    print("\nMissing Values (Top 10):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(10))
    
    print("\nData Types:")
    print(df.dtypes.value_counts())

if __name__ == "__main__":
    analyze_application_data()
