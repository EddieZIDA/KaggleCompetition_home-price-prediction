"""Feature engineering and preprocessing as scikit-learn transformers.

Everything that learns something from data (neighbourhood medians, skewness,
imputation values, one-hot categories, scaling) is fitted on the training fold
only, then reused unchanged on validation/test data. This fixes the train/test
encoding mismatch of the previous version, where the test set had its own
OneHotEncoder and its own imputation statistics.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# In this dataset a missing value in these columns means "the house has none"
# (see data/data_description.txt), not "unknown".
NONE_CATEGORICAL = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
ZERO_NUMERIC = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]
# Numeric codes that are really categories.
AS_CATEGORY = ["MSSubClass", "MoSold"]
# Near-constant: a single train house differs from the rest.
DROP = ["Utilities"]

_QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    **{col: _QUALITY for col in [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
    ]},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}

PORCH_COLS = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]


def _col(data: pd.DataFrame, name: str) -> pd.Series:
    """Column if present, else zeros (keeps the transformer usable on partial frames)."""
    return data[name] if name in data.columns else pd.Series(0, index=data.index)


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain cleaning and feature creation on the raw Kaggle DataFrame.

    Learned state: median LotFrontage per Neighborhood (fallback: global median).
    """

    def fit(self, X: pd.DataFrame, y=None):
        if {"LotFrontage", "Neighborhood"} <= set(X.columns):
            self.lot_frontage_by_nbhd_ = X.groupby("Neighborhood")["LotFrontage"].median().to_dict()
            self.lot_frontage_median_ = float(X["LotFrontage"].median())
        else:
            self.lot_frontage_by_nbhd_, self.lot_frontage_median_ = {}, 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        data = data.drop(columns=[c for c in DROP if c in data.columns])

        for col in NONE_CATEGORICAL:
            if col in data.columns:
                data[col] = data[col].fillna("None")
        for col in ZERO_NUMERIC:
            if col in data.columns:
                data[col] = data[col].fillna(0)

        if "LotFrontage" in data.columns and "Neighborhood" in data.columns:
            fallback = data["Neighborhood"].map(self.lot_frontage_by_nbhd_).fillna(self.lot_frontage_median_)
            data["LotFrontage"] = data["LotFrontage"].fillna(fallback)

        if {"GarageYrBlt", "YearBuilt"} <= set(data.columns):
            # Missing = no garage; the test set also contains a typo (2207).
            bad = data["GarageYrBlt"].isna() | (data["GarageYrBlt"] > 2010)
            data.loc[bad, "GarageYrBlt"] = data.loc[bad, "YearBuilt"]

        for col, mapping in ORDINAL_MAPS.items():
            if col in data.columns:
                # Unseen/missing levels become NaN and are imputed downstream.
                data[col] = data[col].map(mapping).astype(float)

        for col in AS_CATEGORY:
            if col in data.columns:
                data[col] = data[col].astype(str)

        return self._add_features(data)

    @staticmethod
    def _add_features(data: pd.DataFrame) -> pd.DataFrame:
        total_sf = _col(data, "TotalBsmtSF") + _col(data, "1stFlrSF") + _col(data, "2ndFlrSF")
        yr_sold = _col(data, "YrSold")
        new = {
            "TotalSF": total_sf,
            "TotalBathrooms": (_col(data, "FullBath") + 0.5 * _col(data, "HalfBath")
                               + _col(data, "BsmtFullBath") + 0.5 * _col(data, "BsmtHalfBath")),
            "TotalPorchSF": sum(_col(data, c) for c in PORCH_COLS),
            "HouseAge": (yr_sold - _col(data, "YearBuilt")).clip(lower=0),
            "RemodAge": (yr_sold - _col(data, "YearRemodAdd")).clip(lower=0),
            "IsRemodeled": (_col(data, "YearRemodAdd") != _col(data, "YearBuilt")).astype(int),
            "IsNew": (yr_sold == _col(data, "YearBuilt")).astype(int),
            "HasGarage": (_col(data, "GarageArea") > 0).astype(int),
            "HasBsmt": (_col(data, "TotalBsmtSF") > 0).astype(int),
            "Has2ndFlr": (_col(data, "2ndFlrSF") > 0).astype(int),
            "HasPool": (_col(data, "PoolArea") > 0).astype(int),
            "HasFireplace": (_col(data, "Fireplaces") > 0).astype(int),
            "OverallScore": _col(data, "OverallQual") * _col(data, "OverallCond"),
            "QualTotalSF": _col(data, "OverallQual") * total_sf,
            "QualGrLivArea": _col(data, "OverallQual") * _col(data, "GrLivArea"),
        }
        return pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)


class SkewCorrector(BaseEstimator, TransformerMixin):
    """Apply log1p to non-negative numeric columns whose train skewness exceeds a threshold."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            # Constant columns have an undefined skew (NaN) and are left untouched.
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = np.nan_to_num(skew(X, axis=0))
        self.mask_ = (np.abs(skewness) > self.threshold) & (X.min(axis=0) >= 0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.mask_] = np.log1p(np.clip(X[:, self.mask_], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3,
                                 sparse_output=False)),
    ])


def build_preprocessor(kind: str = "tree") -> Pipeline:
    """Full preprocessing: domain features + column-wise imputation/encoding.

    kind="linear" additionally corrects skewness and scales numeric features,
    which linear models, SVR and kernels need; tree models do not.
    """
    if kind == "linear":
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("skew", SkewCorrector()),
            ("scale", RobustScaler()),
        ])
    elif kind == "tree":
        numeric = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind!r}")

    columns = ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", _categorical_pipeline(), make_column_selector(dtype_exclude=np.number)),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("engineer", HouseFeatureEngineer()), ("columns", columns)])
