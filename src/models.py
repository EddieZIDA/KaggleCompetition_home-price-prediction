"""Model zoo: every model is a full Pipeline (preprocessing + estimator).

Because preprocessing lives inside the pipeline, cross-validation refits it on
each training fold, so no statistic from the validation fold can leak.
Tuned hyper-parameters, when present in models/params/<name>.json, override
the defaults below.
"""

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.config import PARAMS_DIR, RANDOM_SEED
from src.features import build_preprocessor

# name -> (preprocessor kind, estimator class, default params)
MODEL_SPECS = {
    "linear": ("linear", LinearRegression, {}),
    "ridge": ("linear", RidgeCV, {"alphas": np.logspace(-1, 3, 40)}),
    "lasso": ("linear", Lasso, {"alpha": 5e-4, "max_iter": 50_000, "random_state": RANDOM_SEED}),
    "elasticnet": ("linear", ElasticNet, {"alpha": 8e-4, "l1_ratio": 0.5, "max_iter": 50_000,
                                          "random_state": RANDOM_SEED}),
    "svr": ("linear", SVR, {"C": 20.0, "epsilon": 0.008, "gamma": 3e-4}),
    "random_forest": ("tree", RandomForestRegressor, {"n_estimators": 500, "max_features": 0.33,
                                                      "min_samples_leaf": 2, "n_jobs": -1,
                                                      "random_state": RANDOM_SEED}),
    "gbr": ("tree", GradientBoostingRegressor, {"n_estimators": 2000, "learning_rate": 0.03,
                                                "max_depth": 4, "max_features": "sqrt",
                                                "min_samples_leaf": 15, "min_samples_split": 10,
                                                "loss": "huber", "subsample": 0.8,
                                                "random_state": RANDOM_SEED}),
    "xgboost": ("tree", XGBRegressor, {"n_estimators": 2000, "learning_rate": 0.02, "max_depth": 3,
                                       "subsample": 0.7, "colsample_bytree": 0.5,
                                       "min_child_weight": 2, "reg_lambda": 1.0, "n_jobs": -1,
                                       "random_state": RANDOM_SEED}),
    "lightgbm": ("tree", LGBMRegressor, {"n_estimators": 3000, "learning_rate": 0.01,
                                         "num_leaves": 6, "max_depth": -1, "min_child_samples": 10,
                                         # subsample only works when subsample_freq > 0
                                         "subsample": 0.75, "subsample_freq": 1,
                                         "colsample_bytree": 0.4, "n_jobs": -1, "verbose": -1,
                                         "random_state": RANDOM_SEED}),
    "catboost": ("tree", CatBoostRegressor, {"iterations": 3000, "learning_rate": 0.03, "depth": 5,
                                             "l2_leaf_reg": 3.0, "verbose": 0, "thread_count": -1,
                                             "allow_writing_files": False,
                                             "random_seed": RANDOM_SEED}),
}

# Models used in the final ensemble (the plain linear regression and the random
# forest are kept as reference baselines only).
ENSEMBLE_CANDIDATES = ["ridge", "lasso", "elasticnet", "svr", "gbr", "xgboost", "lightgbm", "catboost"]


def params_path(name: str, params_dir: Path = PARAMS_DIR) -> Path:
    return params_dir / f"{name}.json"


def load_params(name: str, params_dir: Path = PARAMS_DIR) -> dict:
    """Default params updated with tuned ones if a JSON file exists."""
    _, _, defaults = MODEL_SPECS[name]
    params = dict(defaults)
    path = params_path(name, params_dir)
    if path.exists():
        params.update(json.loads(path.read_text(encoding="utf-8")))
    return params


def save_params(name: str, params: dict, params_dir: Path = PARAMS_DIR) -> Path:
    params_dir.mkdir(parents=True, exist_ok=True)
    path = params_path(name, params_dir)
    path.write_text(json.dumps(params, indent=4), encoding="utf-8")
    return path


def build_model(name: str, params: dict | None = None) -> Pipeline:
    """Preprocessing + estimator for `name`, with tuned params unless `params` is given."""
    kind, estimator_cls, _ = MODEL_SPECS[name]
    params = load_params(name) if params is None else {**MODEL_SPECS[name][2], **params}
    return Pipeline([("prep", build_preprocessor(kind)), ("model", estimator_cls(**params))])
