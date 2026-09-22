import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).parent
INGESTED_FILE = BASE_DIR / "ingested" / "B.csv"

def split_data():
    df = pd.read_csv(INGESTED_FILE)
    df.drop(columns=['student_id'], inplace=True)

    y_stratify = df['placement_status']

    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_stratify
    )

    train_df.to_csv(BASE_DIR / "train.csv", index=False)
    test_df.to_csv(BASE_DIR / "test.csv", index=False)
    print(f"Split selesai -> Train: {train_df.shape[0]}, Test: {test_df.shape[0]}")

if __name__ == "__main__":
    split_data()