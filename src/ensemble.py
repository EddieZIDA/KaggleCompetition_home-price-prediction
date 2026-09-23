"""Blending and stacking on out-of-fold predictions.

Both ensembles are scored honestly: the blend weights (or the meta-model) used
to score fold k are learned on the out-of-fold predictions of the other folds
only, so the reported ensemble score is not fitted on its own validation data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from src.data import partial_sale_mansions, to_price
from src.evaluation import make_cv, rmse


@dataclass
class EnsembleRegressor:
    """Fitted base pipelines combined by blend weights or a linear meta-model.

    Predicts log1p(SalePrice) with `predict` and dollars with `predict_price`.
    When `mansion_log_price` is set, houses matching `partial_sale_mansions` get
    that value (learned from their training-set twins) instead of the model output.
    """

    method: str                      # "blend" or "stack"
    models: dict = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    meta_model: LinearRegression | None = None
    mansion_log_price: float | None = None

    def base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([model.predict(X) for model in self.models.values()])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = self.base_predictions(X)
        if self.method == "stack":
            predictions = self.meta_model.predict(base)
        else:
            predictions = base @ np.array([self.weights[name] for name in self.models])
        if self.mansion_log_price is not None:
            predictions[partial_sale_mansions(X).to_numpy()] = self.mansion_log_price
        return predictions

    def predict_price(self, X: pd.DataFrame) -> np.ndarray:
        return to_price(self.predict(X))


def optimize_weights(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise the RMSE of oof @ w."""
    n_models = oof.shape[1]
    result = minimize(
        lambda w: rmse(y, oof @ w),
        x0=np.full(n_models, 1 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    weights = np.clip(result.x, 0, None)
    return weights / weights.sum()


def blend_cv_scores(oof: np.ndarray, y: np.ndarray, cv: KFold | None = None) -> np.ndarray:
    """Fold scores of the blend, with weights learned without the scored fold."""
    cv = cv or make_cv()
    scores = []
    for train_idx, val_idx in cv.split(oof):
        weights = optimize_weights(oof[train_idx], y[train_idx])
        scores.append(rmse(y[val_idx], oof[val_idx] @ weights))
    return np.array(scores)


def make_meta_model() -> LinearRegression:
    # Positive coefficients keep the stack interpretable and stop it from
    # exploiting the strong correlation between base models.
    return LinearRegression(positive=True)


def stack_cv_scores(oof: np.ndarray, y: np.ndarray, cv: KFold | None = None) -> np.ndarray:
    """Fold scores of a linear meta-model trained on the other folds' out-of-fold predictions."""
    cv = cv or make_cv()
    scores = []
    for train_idx, val_idx in cv.split(oof):
        meta = make_meta_model().fit(oof[train_idx], y[train_idx])
        scores.append(rmse(y[val_idx], meta.predict(oof[val_idx])))
    return np.array(scores)
