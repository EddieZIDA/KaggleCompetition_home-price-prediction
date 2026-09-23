"""End-to-end training pipeline.

    python -m src.pipeline                   # evaluate, ensemble, write submission
    python -m src.pipeline --tune --trials 40
    python -m src.pipeline --models ridge lasso xgboost --no-save

Steps: load data -> (optional) Optuna tuning -> 5-fold CV of every model ->
honest blend/stack evaluation -> refit on the full training set -> validated
submission + metrics, figures and the fitted ensemble on disk.
"""

import argparse
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd

from src import visualization as viz
from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SUBMISSIONS_DIR
from src.data import load_test, load_train
from src.ensemble import (
    EnsembleRegressor,
    blend_cv_scores,
    make_meta_model,
    optimize_weights,
    stack_cv_scores,
)
from src.evaluation import CVResult, cross_validate, results_table
from src.models import ENSEMBLE_CANDIDATES, MODEL_SPECS, build_model
from src.submission import write_submission
from src.tuning import SEARCH_SPACES, tune

BASELINES = ["linear", "random_forest"]
# Honest 5-fold CV of the previous pipeline's XGB+LGB+Ridge blend, measured
# during the audit (its own submission file was in log space and unusable).
PREVIOUS_PIPELINE_RMSLE = 0.1204

warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass
class PipelineResult:
    table: pd.DataFrame
    cv_results: dict[str, CVResult]
    ensemble: EnsembleRegressor
    blend_scores: np.ndarray
    stack_scores: np.ndarray
    submission: pd.DataFrame | None


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _feature_importance(ensemble: EnsembleRegressor) -> tuple[str, pd.Series] | None:
    """Normalised gain importance of the first boosted-tree model kept in the ensemble."""
    for name in ("xgboost", "lightgbm", "catboost"):
        if name not in ensemble.models:
            continue
        pipeline = ensemble.models[name]
        names = pipeline.named_steps["prep"].named_steps["columns"].get_feature_names_out()
        model = pipeline.named_steps["model"]
        if name == "xgboost":
            gain = np.zeros(len(names))
            for feature, value in model.get_booster().get_score(importance_type="gain").items():
                gain[int(feature[1:])] = value      # booster features are named f0, f1, ...
        elif name == "lightgbm":
            gain = model.booster_.feature_importance(importance_type="gain")
        else:
            gain = model.get_feature_importance()
        return name, pd.Series(gain / gain.sum(), index=names)
    return None


def _package_versions() -> dict:
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna"]
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}


def run(models: list[str] | None = None, tune_models: bool = False, n_trials: int = 40,
        tune_timeout: float | None = None, save: bool = True, verbose: bool = True) -> PipelineResult:
    models = models or BASELINES + ENSEMBLE_CANDIDATES
    X, y = load_train()
    X_test, test_ids = load_test()
    _log(f"Train: {X.shape[0]} houses x {X.shape[1]} raw features | Test: {X_test.shape[0]} houses",
         verbose)

    if tune_models:
        for name in [m for m in models if m in SEARCH_SPACES]:
            _log(f"Tuning {name} ({n_trials} trials)...", verbose)
            params, score = tune(name, X, y, n_trials=n_trials, timeout=tune_timeout)
            _log(f"  best tuning-fold RMSLE {score:.4f}  {params}", verbose)

    _log("\nCross-validation (5 folds, preprocessing refitted inside each fold)", verbose)
    cv_results = {}
    for name in models:
        cv_results[name] = cross_validate(build_model(name), X, y, name)
        _log(f"  {cv_results[name]}", verbose)

    members = [m for m in models if m in ENSEMBLE_CANDIDATES]
    oof = np.column_stack([cv_results[m].oof for m in members])
    y_arr = y.to_numpy()
    blend_scores = blend_cv_scores(oof, y_arr)
    stack_scores = stack_cv_scores(oof, y_arr)
    weights = dict(zip(members, optimize_weights(oof, y_arr), strict=True))
    blend_oof = oof @ np.array(list(weights.values()))
    _log(f"  {'ensemble_blend':15s} RMSLE {blend_scores.mean():.4f} ± {blend_scores.std():.4f}", verbose)
    _log(f"  {'ensemble_stack':15s} RMSLE {stack_scores.mean():.4f} ± {stack_scores.std():.4f}", verbose)

    table = results_table(list(cv_results.values()))
    ensembles = pd.DataFrame([
        {"model": "ensemble_blend", "rmsle_mean": blend_scores.mean(), "rmsle_std": blend_scores.std()},
        {"model": "ensemble_stack", "rmsle_mean": stack_scores.mean(), "rmsle_std": stack_scores.std()},
    ])
    table = pd.concat([table, ensembles], ignore_index=True).sort_values("rmsle_mean", ignore_index=True)

    method = "blend" if blend_scores.mean() <= stack_scores.mean() else "stack"
    kept = [m for m in members if method == "stack" or weights[m] > 1e-3]
    _log(f"\nSelected: {method} over {kept}. Refitting on the full training set...", verbose)
    ensemble = EnsembleRegressor(method=method)
    for name in kept:
        ensemble.models[name] = build_model(name).fit(X, y)
    if method == "blend":
        total = sum(weights[m] for m in kept)
        ensemble.weights = {m: weights[m] / total for m in kept}
    else:
        ensemble.meta_model = make_meta_model().fit(oof, y_arr)

    submission = None
    if save:
        submission = write_submission(test_ids, ensemble.predict_price(X_test),
                                      SUBMISSIONS_DIR / "submission.csv")
        _save_artifacts(ensemble, table, cv_results, members, y, blend_oof, weights, method)
        _log(f"Submission written: {SUBMISSIONS_DIR / 'submission.csv'} "
             f"({len(submission)} rows, median price ${submission.SalePrice.median():,.0f})", verbose)

    return PipelineResult(table, cv_results, ensemble, blend_scores, stack_scores, submission)


def _save_artifacts(ensemble, table, cv_results, members, y, blend_oof, weights, method) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, MODELS_DIR / "final_ensemble.joblib")

    table.to_csv(RESULTS_DIR / "cv_results.csv", index=False, float_format="%.5f")
    oof_frame = pd.DataFrame({name: r.oof for name, r in cv_results.items()})
    oof_frame.insert(0, "y_log", y.to_numpy())
    oof_frame.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False, float_format="%.6f")

    best = table.iloc[0]
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": "RMSLE (RMSE of log1p(SalePrice)), 5-fold CV",
        "selected_method": method,
        "best": {"model": best["model"], "rmsle_mean": round(float(best["rmsle_mean"]), 5),
                 "rmsle_std": round(float(best["rmsle_std"]), 5)},
        "previous_pipeline_rmsle": PREVIOUS_PIPELINE_RMSLE,
        "blend_weights": {k: round(float(v), 4) for k, v in weights.items()},
        "models": {row.model: {"rmsle_mean": round(row.rmsle_mean, 5), "rmsle_std": round(row.rmsle_std, 5)}
                   for row in table.itertuples()},
        "n_train": int(len(y)),
        "environment": _package_versions(),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    viz.plot_model_comparison(table, FIGURES_DIR / "model_comparison.png", PREVIOUS_PIPELINE_RMSLE)
    viz.plot_predictions(y, blend_oof, FIGURES_DIR / "oof_predictions.png")
    viz.plot_blend_weights(weights, FIGURES_DIR / "blend_weights.png")
    importance = _feature_importance(ensemble)
    if importance is not None:
        model_name, values = importance
        viz.plot_feature_importance(values, FIGURES_DIR / "feature_importance.png", model_name)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # figures are only saved from the command line
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), help="models to evaluate")
    parser.add_argument("--tune", action="store_true", help="run Optuna before evaluating")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials per tuned model")
    parser.add_argument("--tune-timeout", type=float, help="max seconds of tuning per model")
    parser.add_argument("--no-save", action="store_true", help="do not write submission/artifacts")
    args = parser.parse_args()
    result = run(args.models, args.tune, args.trials, args.tune_timeout, save=not args.no_save)
    print("\n" + result.table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
