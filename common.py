"""Shared data loading, splitting and feature engineering.

Used by notebook.ipynb, train.py, predict.py and predict_test.py so all of them
prepare data in exactly the same way. `add_features` is also part of the saved
model pipeline, which is why this module must be importable wherever the model
is loaded (including the Docker image).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "dataset" / "creditcard.csv.gz"
TARGET = "Class"
SEED = 42

# Columns of a raw transaction, as sent to the web service.
RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Columns the model is trained on (output of add_features).
FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows.

    Done before splitting so the same transaction can never end up in both the
    training and the test set (which would inflate test scores).
    """
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int = SEED):
    """Stratified 60/20/20 train/validation/test split.

    Stratified so each part keeps the 0.17% fraud rate.
    """
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=seed
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, stratify=df_full_train[TARGET], random_state=seed
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction columns into model features.

    - Time is seconds since the first transaction in the dataset, which means
      nothing for a new transaction. Only the time of day is kept (`hour`).
    - Amount is heavily right-skewed, so it is log-transformed (`log_amount`).
    """
    out = df[[f"V{i}" for i in range(1, 29)]].copy()
    out["log_amount"] = np.log1p(df["Amount"])
    out["hour"] = (df["Time"] % 86_400) / 3_600
    return out[FEATURES]
