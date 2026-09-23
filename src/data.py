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


def partial_sale_mansions(houses: pd.DataFrame) -> pd.Series:
    """Very large Edwards houses sold as "Partial" (new construction sold before completion).

    Both such houses in train sold far below market value (160k and 185k dollars for
    4,700-5,600 sq ft) and are removed as outliers, so no model can learn them. The
    test set contains one more (Id 2550); models extrapolate it to ~865k dollars.
    """
    required = {"GrLivArea", "Neighborhood", "SaleCondition"}
    if not required <= set(houses.columns):
        return pd.Series(False, index=houses.index)
    return ((houses["GrLivArea"] > 4000) & (houses["Neighborhood"] == "Edwards")
            & (houses["SaleCondition"] == "Partial"))


def partial_sale_mansion_log_price(path: Path = TRAIN_PATH) -> float | None:
    """Mean log1p price of the partial-sale mansions of the raw training set."""
    train = pd.read_csv(path)
    mask = partial_sale_mansions(train)
    return float(np.log1p(train.loc[mask, TARGET]).mean()) if mask.any() else None


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
