import pandas as pd
from pathlib import Path

print("dataload.py loaded")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Data"/"data" / "diabetes.csv"

def read_data():
    """Read the diabetes.csv file from the data folder at the project root."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    return pd.read_csv(DATA_FILE)

if __name__ == "__main__":
    df = read_data()
    print(df.head())