"""
plot_utils.py
-------------
Publication-friendly plotting for the noise study.

Three main plots
----------------
1. performance_vs_shots    — RMSE / R² vs shots (shot-noise sweep)
2. noise_type_comparison   — bar/point chart comparing noise types at fixed shots
3. prediction_trace        — example ground truth vs prediction traces
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

NOISE_ORDER = [
    "ideal",
    "shot_only",
    "readout_only",
    "single_qubit_only",
    "two_qubit_only",
    "relaxation_only",
    "full_backend",
]

NOISE_LABELS = {
    "ideal": "Ideal",
    "shot_only": "Shot only",
    "readout_only": "Readout",
    "single_qubit_only": "1Q gate",
    "two_qubit_only": "2Q gate",
    "relaxation_only": "Relaxation",
    "full_backend": "Full backend",
}

PALETTE = sns.color_palette("tab10", n_colors=len(NOISE_ORDER))
COLOR_MAP = dict(zip(NOISE_ORDER, PALETTE))


def _apply_style():
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# 1. Performance vs shots (shot-noise sweep)
# ---------------------------------------------------------------------------

def plot_performance_vs_shots(
    master_df: pd.DataFrame,
    metric: str = "rmse",
    figures_dir: Path = Path("results/figures"),
) -> Path:
    """
    Line plot: metric (mean ± std across seeds) vs log-scale shots.
    Only uses rows where experiment_type == 'shot_sweep'.

    Returns path to saved figure.
    """
    _apply_style()
    df = master_df[master_df["experiment_type"] == "shot_sweep"].copy()
    if df.empty:
        logger.warning("No shot_sweep rows found in master_df.")
        return Path()

    grp = (
        df.groupby("shots")[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        grp["shots"], grp["mean"], yerr=grp["std"],
        marker="o", linewidth=1.8, capsize=4,
        color=COLOR_MAP.get("shot_only", "steelblue"),
        label="Shot-only noise",
    )

    # Add ideal reference if present
    ideal_rows = master_df[master_df["noise_type"] == "ideal"]
    if not ideal_rows.empty:
        ideal_mean = ideal_rows[metric].mean()
        ax.axhline(ideal_mean, linestyle="--", color="black",
                   linewidth=1.2, label="Ideal (noiseless)")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Shots", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Effect of Shot Noise: {metric.upper()} vs Shots", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"performance_vs_shots_{metric}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 2. Noise-type comparison (ablation)
# ---------------------------------------------------------------------------

def plot_noise_type_comparison(
    master_df: pd.DataFrame,
    metric: str = "rmse",
    figures_dir: Path = Path("results/figures"),
) -> Path:
    """
    Point plot with error bars: metric per noise type at fixed shots (ablation).
    Includes ideal baseline for reference.

    Returns path to saved figure.
    """
    _apply_style()

    ablation_df = master_df[
        (master_df["experiment_type"] == "ablation") |
        (master_df["noise_type"] == "ideal")
    ].copy()

    if ablation_df.empty:
        logger.warning("No ablation rows found in master_df.")
        return Path()

    grp = (
        ablation_df.groupby("noise_type")[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Sort by NOISE_ORDER
    grp["order"] = grp["noise_type"].map(
        {k: i for i, k in enumerate(NOISE_ORDER)}
    )
    grp = grp.sort_values("order").reset_index(drop=True)
    grp["label"] = grp["noise_type"].map(NOISE_LABELS)
    grp["color"] = grp["noise_type"].map(COLOR_MAP)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        grp["label"], grp["mean"], xerr=grp["std"],
        color=grp["color"].tolist(), edgecolor="black", linewidth=0.6,
        capsize=4, height=0.55,
    )
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Noise-Type Ablation: {metric.upper()} @ 1024 shots", fontsize=13)
    ax.invert_yaxis()
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"noise_type_comparison_{metric}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 3. Prediction trace
# ---------------------------------------------------------------------------

def plot_prediction_trace(
    predictions_csv: Path,
    run_id: str,
    figures_dir: Path = Path("results/figures"),
    n_steps: int = 300,
) -> Path:
    """
    Plot ground truth vs predicted trace for a single run.

    Parameters
    ----------
    predictions_csv : Path
        Path to predictions.csv for the chosen run.
    run_id : str
        Label for the plot title and filename.
    n_steps : int
        How many time steps to display.
    """
    _apply_style()
    df = pd.read_csv(predictions_csv)
    df = df.iloc[:n_steps]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df.index, df["y_true"], label="Ground truth", color="black",
            linewidth=1.2, alpha=0.85)
    ax.plot(df.index, df["y_pred"], label="Prediction", color="steelblue",
            linewidth=1.2, linestyle="--", alpha=0.9)

    ax.set_xlabel("Test step", fontsize=11)
    ax.set_ylabel("Amplitude (normalised)", fontsize=11)
    ax.set_title(f"Prediction trace — {run_id}", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    safe_id = run_id.replace("/", "_")
    out = figures_dir / f"prediction_trace_{safe_id}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def make_all_plots(
    master_df: pd.DataFrame,
    runs_root: Path,
    figures_dir: Path,
    trace_run_id: Optional[str] = None,
) -> None:
    """
    Generate all standard plots from master results DataFrame.

    Parameters
    ----------
    trace_run_id : str or None
        Specific run_id for the prediction trace.  If None, picks the
        first successful ideal run automatically.
    """
    for metric in ("rmse", "r2"):
        plot_performance_vs_shots(master_df, metric=metric,
                                  figures_dir=figures_dir)
        plot_noise_type_comparison(master_df, metric=metric,
                                   figures_dir=figures_dir)

    # Prediction trace
    if trace_run_id is None:
        ideal_rows = master_df[master_df["noise_type"] == "ideal"]
        if not ideal_rows.empty:
            trace_run_id = ideal_rows.iloc[0]["run_id"]

    if trace_run_id:
        pred_csv = runs_root / trace_run_id / "predictions.csv"
        if pred_csv.exists():
            plot_prediction_trace(pred_csv, trace_run_id,
                                  figures_dir=figures_dir)
