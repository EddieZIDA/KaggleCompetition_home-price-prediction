"""Static figures for the README and the notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import to_price

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
ACCENT = "#2a78d6"      # categorical slot 1
SECONDARY = "#eb6834"   # categorical slot 2
NEUTRAL = "#b9b8b2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(table: pd.DataFrame, path: Path, baseline: float | None = None) -> Path:
    """Horizontal bars, best at the top; ensembles in the accent colour, single models muted."""
    table = table.sort_values("rmsle_mean", ascending=False).reset_index(drop=True)
    x_min, x_max = 0.095, 0.145     # zoom on the competitive range; worse models are marked off-scale
    is_ensemble = table["model"].str.startswith("ensemble")
    colors = np.where(is_ensemble, ACCENT, NEUTRAL)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(table) + 1.6))
    shown = table["rmsle_mean"].clip(upper=x_max)
    ax.barh(table["model"], shown - x_min, left=x_min, color=colors, height=0.6)
    for y, row in table.iterrows():
        off_scale = row.rmsle_mean >= x_max
        if not off_scale:
            ax.errorbar(row.rmsle_mean, y, xerr=row.rmsle_std, color=TEXT_MUTED, elinewidth=1, capsize=2)
        # Scores in a right-aligned column, like a table, so they never collide with the marks.
        label = f"{row.rmsle_mean:.4f} ± {row.rmsle_std:.4f}" + ("  (off scale)" if off_scale else "")
        ax.text(x_max + 0.0012, y, label, va="center", ha="left", fontsize=9,
                color=TEXT, fontweight="bold" if row.model.startswith("ensemble") else "normal")
    if baseline is not None:
        ax.axvline(baseline, color=SECONDARY, linewidth=1.5, linestyle="--")
        ax.text(baseline + 0.0005, -0.9, f"previous version {baseline:.4f}", color=TEXT_MUTED,
                fontsize=8.5, va="center")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.3, len(table) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSLE, 5-fold CV (lower is better)")
    ax.set_title("Model comparison — out-of-fold RMSLE")
    return _save(fig, path)


def plot_predictions(y_log: pd.Series, oof_log: np.ndarray, path: Path) -> Path:
    """Out-of-fold predicted vs actual price, plus the residual distribution (log scale)."""
    actual, predicted = to_price(y_log.to_numpy()), to_price(oof_log)
    residuals = y_log.to_numpy() - oof_log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    ax1.scatter(actual / 1e3, predicted / 1e3, s=10, color=ACCENT, alpha=0.45, linewidths=0)
    lims = [0, max(actual.max(), predicted.max()) / 1e3 * 1.03]
    ax1.plot(lims, lims, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax1.set(xlim=lims, ylim=lims, xlabel="Actual price (k$)", ylabel="Predicted price (k$)",
            title="Out-of-fold predictions vs actual")

    ax2.hist(residuals, bins=60, color=ACCENT, edgecolor=SURFACE, linewidth=0.6)
    ax2.axvline(0, color=TEXT_MUTED, linewidth=1)
    ax2.set(xlabel="Residual  log(actual) − log(predicted)", ylabel="Houses",
            title="Residual distribution")
    ax2.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_blend_weights(weights: dict[str, float], path: Path) -> Path:
    series = pd.Series(weights).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(series) + 1.2))
    ax.barh(series.index, series.values, color=ACCENT, height=0.6)
    for y, w in enumerate(series.values):
        ax.text(w + 0.005, y, f"{w:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, series.max() * 1.18 + 0.01)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Weight in the blend")
    ax.set_title("Optimised blend weights")
    return _save(fig, path)


def plot_feature_importance(importance: pd.Series, path: Path, model_name: str = "xgboost",
                            top: int = 20) -> Path:
    series = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(series) + 1.2))
    ax.barh(series.index, series.values, color=ACCENT, height=0.65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total gain")
    ax.set_title(f"Top {top} features ({model_name} gain)")
    return _save(fig, path)
