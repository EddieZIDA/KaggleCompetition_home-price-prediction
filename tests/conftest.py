import numpy as np
import pandas as pd
import pytest

from src.config import TEST_PATH, TRAIN_PATH


def make_raw_houses(n: int, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the same kinds of columns (and quirks) as the Kaggle data."""
    rng = np.random.default_rng(seed)
    year_built = rng.integers(1900, 2009, n)
    has_garage = rng.random(n) > 0.1
    has_fireplace = rng.random(n) > 0.5
    frame = pd.DataFrame({
        "MSSubClass": rng.choice([20, 60, 120], n),
        "MSZoning": rng.choice(["RL", "RM", "FV"], n),
        "LotFrontage": np.where(rng.random(n) > 0.2, rng.integers(40, 120, n), np.nan),
        "LotArea": rng.lognormal(9, 0.5, n).round(),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown"], n),
        "Heating": rng.choice(["GasA", "GasW"], n, p=[0.95, 0.05]),
        "OverallQual": rng.integers(1, 11, n),
        "OverallCond": rng.integers(1, 10, n),
        "YearBuilt": year_built,
        "YearRemodAdd": year_built + rng.integers(0, 10, n),
        "TotalBsmtSF": rng.integers(0, 2000, n).astype(float),
        "1stFlrSF": rng.integers(500, 2000, n),
        "2ndFlrSF": rng.choice([0, 600, 900], n),
        "GrLivArea": rng.integers(600, 3500, n),
        "FullBath": rng.integers(1, 4, n),
        "HalfBath": rng.integers(0, 2, n),
        "BsmtFullBath": rng.integers(0, 2, n).astype(float),
        "BsmtHalfBath": rng.integers(0, 2, n).astype(float),
        "KitchenQual": rng.choice(["TA", "Gd", "Ex"], n),
        "Fireplaces": has_fireplace.astype(int),
        "FireplaceQu": np.where(has_fireplace, rng.choice(["TA", "Gd"], n), None),
        "GarageType": np.where(has_garage, "Attchd", None),
        "GarageYrBlt": np.where(has_garage, year_built, np.nan),
        "GarageArea": np.where(has_garage, rng.integers(200, 900, n), np.nan),
        "PoolArea": np.zeros(n, dtype=int),
        "OpenPorchSF": rng.integers(0, 100, n),
        "WoodDeckSF": rng.integers(0, 300, n),
        "MoSold": rng.integers(1, 13, n),
        "YrSold": rng.integers(2006, 2011, n),
        "Utilities": "AllPub",
    })
    frame["YearRemodAdd"] = frame[["YearRemodAdd", "YrSold"]].min(axis=1)
    return frame


@pytest.fixture
def raw_train() -> pd.DataFrame:
    return make_raw_houses(200, seed=0)


@pytest.fixture
def raw_test() -> pd.DataFrame:
    return make_raw_houses(80, seed=1)


@pytest.fixture
def synthetic_target(raw_train) -> pd.Series:
    price = 20_000 + 60 * raw_train["GrLivArea"] + 8_000 * raw_train["OverallQual"]
    return np.log1p(price.astype(float))


requires_kaggle_data = pytest.mark.skipif(
    not (TRAIN_PATH.exists() and TEST_PATH.exists()),
    reason="Kaggle CSV files not found in data/raw",
)
