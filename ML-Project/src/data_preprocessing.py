"""Data loading and train/test splitting for the Iris classifier."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "IRIS.csv"
TARGET_COLUMN = "species"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(file_path=DATA_PATH):
    """Read a CSV dataset and report a useful error if it is unavailable."""
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def preprocess_data(df, target_column=TARGET_COLUMN):
    """Validate a labeled dataset and return a reproducible stratified split."""
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    if df[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    if X.empty or not len(X.columns):
        raise ValueError("The dataset must contain at least one feature column.")

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print(f"Loaded {len(data)} rows with {len(data.columns) - 1} features.")
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}.")
