"""Cross-validation helpers. Scores are RMSE on log1p(SalePrice), i.e. the Kaggle RMSLE."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from src.config import CV_SEED, N_FOLDS


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def make_cv(seed: int = CV_SEED, n_folds: int = N_FOLDS) -> KFold:
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


@dataclass
class CVResult:
    name: str
    fold_scores: np.ndarray
    oof: np.ndarray
    fit_seconds: float

    @property
    def mean(self) -> float:
        return float(self.fold_scores.mean())

    @property
    def std(self) -> float:
        return float(self.fold_scores.std())

    def __str__(self) -> str:
        return f"{self.name:15s} RMSLE {self.mean:.4f} ± {self.std:.4f}  ({self.fit_seconds:.0f}s)"


def cross_validate(model, X: pd.DataFrame, y: pd.Series, name: str = "model",
                   cv: KFold | None = None) -> CVResult:
    """Out-of-fold predictions and per-fold scores; the pipeline is refitted on every fold."""
    cv = cv or make_cv()
    oof = np.zeros(len(y))
    scores = []
    start = time.perf_counter()
    for train_idx, val_idx in cv.split(X):
        fold_model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fold_model.predict(X.iloc[val_idx])
        scores.append(rmse(y.iloc[val_idx], oof[val_idx]))
    return CVResult(name, np.array(scores), oof, time.perf_counter() - start)


def results_table(results: list[CVResult]) -> pd.DataFrame:
    table = pd.DataFrame(
        {"model": r.name, "rmsle_mean": r.mean, "rmsle_std": r.std, "fit_seconds": r.fit_seconds}
        for r in results
    )
    return table.sort_values("rmsle_mean").reset_index(drop=True)
