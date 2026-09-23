"""Data loading, outlier removal and target transformation.

The target is log-transformed in exactly one place (`load_train`) and brought
back to dollars in exactly one place (`to_price`), so the double-log bug of the
previous version cannot happen again.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET, TEST_PATH, TRAIN_PATH


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop the two huge, cheap houses flagged by the dataset author (De Cock, 2011).

    Only ever applied to the training set: every test row must be predicted.
    """
    mask = (train["GrLivArea"] > 4000) & (train[TARGET] < 300_000)
    return train.loc[~mask].reset_index(drop=True)


def load_train(path: Path = TRAIN_PATH, drop_outliers: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw training features and the log1p-transformed target."""
    train = pd.read_csv(path)
    if drop_outliers:
        train = remove_outliers(train)
    y_log = np.log1p(train[TARGET]).rename("log_SalePrice")
    X = train.drop(columns=[TARGET, ID_COL])
    return X, y_log


def load_test(path: Path = TEST_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw test features and the Kaggle ids."""
    test = pd.read_csv(path)
    return test.drop(columns=[ID_COL]), test[ID_COL]


def to_price(y_log: np.ndarray) -> np.ndarray:
    """Inverse of the log1p target transform."""
    return np.expm1(y_log)
