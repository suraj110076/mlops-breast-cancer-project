import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
)


def ingest_data(test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset, save raw data, and create train/test splits."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df[TARGET_COLUMN] = dataset.target

    df.to_csv(RAW_DATA_PATH, index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    return train_df, test_df
