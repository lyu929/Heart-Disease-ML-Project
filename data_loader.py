import pandas as pd
from pathlib import Path


# ── Change this to switch datasets ──────────────
# Options: "heart", "cleveland", "kaggle", "framingham"
DATASET = "heart"
# ────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_FILES = {
    "heart":      "data/heart.csv",
    "cleveland":  "data/cleveland_processed.csv",
    "kaggle":     "data/kaggle_processed.csv",
    "framingham": "data/framingham_processed.csv",
}

TARGET_COLUMNS = {
    "heart": "HeartDisease",
    "cleveland": "target",
    "kaggle": "target",
    "framingham": "target",
}

DATA_PATH = PROJECT_DIR / DATASET_FILES[DATASET]
TARGET_COL = TARGET_COLUMNS[DATASET]


def get_dataset_name():
    return DATASET


def get_target_column():
    return TARGET_COL


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Please put the selected dataset file into the data/ folder."
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
