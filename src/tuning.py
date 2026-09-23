"""Optuna hyper-parameter search.

Tuning uses its own fold seed (TUNING_CV_SEED) so the reported CV scores come
from splits the optimiser never saw, and a seeded TPE sampler so the search is
reproducible.
"""

import optuna

from src.config import RANDOM_SEED, TUNING_CV_SEED
from src.evaluation import cross_validate, make_cv
from src.models import build_model, save_params


def _space_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 4000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _space_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 5000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, 32),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
    }


def _space_catboost(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 1000, 4000, step=250),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("depth", 3, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def _space_svr(trial: optuna.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1.0, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 1e-2, log=True),
    }


def _space_lasso(trial: optuna.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 5e-5, 5e-3, log=True)}


def _space_elasticnet(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 5e-5, 1e-2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0),
    }


SEARCH_SPACES = {
    "xgboost": _space_xgboost,
    "lightgbm": _space_lightgbm,
    "catboost": _space_catboost,
    "svr": _space_svr,
    "lasso": _space_lasso,
    "elasticnet": _space_elasticnet,
}


def tune(name: str, X, y, n_trials: int = 50, timeout: float | None = None) -> tuple[dict, float]:
    """Search hyper-parameters for `name`, save them to models/params and return (params, score)."""
    space = SEARCH_SPACES[name]
    cv = make_cv(seed=TUNING_CV_SEED)

    def objective(trial: optuna.Trial) -> float:
        return cross_validate(build_model(name, space(trial)), X, y, name, cv).mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    save_params(name, study.best_params)
    return study.best_params, study.best_value
