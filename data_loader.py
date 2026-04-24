import os
import pandas as pd


# ── Change this to switch datasets ──────────────
# Options: "cleveland", "kaggle", "framingham"
DATASET = "cleveland"
# ────────────────────────────────────────────────
 
DATASET_FILES = {
    "cleveland":  "data/cleveland_processed.csv",
    "kaggle":     "data/kaggle_processed.csv",
    "framingham": "data/framingham_processed.csv",
}
 
DATA_PATH  = DATASET_FILES[DATASET]
TARGET_COL = "target"


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Please put heart.csv into the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    return df


def inspect_dataset(df):
    print("Shape:", df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))
