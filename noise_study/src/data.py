"""
data.py
-------
Dataset preparation for the noise study.

I delegate Mackey-Glass generation to my own MG_series() function in
QRC/datasets.py, which is my battle-tested implementation.

If datasets.py is not importable, this module falls back to a self-contained
RK4 generator so the noise study can still run standalone.

Path layout:
    QRC/
        datasets.py             ← my code
        noise_study/
            src/
                data.py         ← this file
    parents[2] of this file = QRC/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap: add QRC/ to sys.path so `from datasets import MG_series` works
# ---------------------------------------------------------------------------
_QRC_ROOT = Path(__file__).resolve().parents[2]   # QRC/noise_study/src → QRC/

if str(_QRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_QRC_ROOT))

try:
    from datasets import MG_series as _my_MG_series
    _USE_MY_DATASET = True
    logger.info("data.py: using MG_series from QRC/datasets.py")
except ImportError as _e:
    _USE_MY_DATASET = False
    logger.warning(
        "data.py: QRC/datasets.py not importable (%s). "
        "Using built-in RK4 fallback.", _e
    )


# ---------------------------------------------------------------------------
# Fallback RK4 Mackey-Glass generator
# ---------------------------------------------------------------------------

def _generate_mackey_glass_rk4(
    n_samples: int,
    tau: int,
    washout: int = 200,
) -> np.ndarray:
    """
    Integrate the Mackey-Glass DDE with RK4 + discrete delay buffer.
    Returns a 1-D array of length n_samples (transient already removed).
    """
    beta, gamma, n_exp, dt = 0.2, 0.1, 10, 1.0
    total = n_samples + washout
    history_len = max(tau + 1, 100)

    x = np.zeros(total + history_len)
    x[:history_len] = 0.9 + 0.05 * np.sin(
        np.linspace(0, 2 * np.pi, history_len)
    )

    def dxdt(xt: float, x_tau: float) -> float:
        return beta * x_tau / (1.0 + x_tau ** n_exp) - gamma * xt

    for t in range(history_len, total + history_len - 1):
        xt, x_tau = x[t], x[t - tau]
        k1 = dxdt(xt, x_tau)
        k2 = dxdt(xt + 0.5 * dt * k1, x_tau)
        k3 = dxdt(xt + 0.5 * dt * k2, x_tau)
        k4 = dxdt(xt + dt * k3, x_tau)
        x[t + 1] = xt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x[history_len + washout:]


# ---------------------------------------------------------------------------
# Windowing helper
# ---------------------------------------------------------------------------

def _create_windows(
    series: np.ndarray,
    window_size: int,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over series.
    X[i] = series[i : i+window_size]
    Y[i] = series[i + window_size + prediction_horizon - 1]  (scalar target)
    """
    X, Y = [], []
    for i in range(len(series) - window_size - prediction_horizon + 1):
        X.append(series[i: i + window_size])
        Y.append(series[i + window_size + prediction_horizon - 1])
    return np.array(X), np.array(Y)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_01(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Min-max normalise to [0, 1]. Returns (normalised, min, max)."""
    a_min, a_max = float(arr.min()), float(arr.max())
    return (arr - a_min) / (a_max - a_min + 1e-12), a_min, a_max


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, paths_cfg: dict, force_rebuild: bool = False) -> dict:
    """
    Build (or load cached) the Mackey-Glass dataset.

    Parameters
    ----------
    cfg       : dataset sub-dict from study_config.yaml
    paths_cfg : data sub-dict from paths.yaml
    force_rebuild : regenerate even if cache exists

    Returns
    -------
    dict:
        X_train, y_train, X_test, y_test  — numpy arrays
        series                            — 1-D normalised series (for plotting)
        s_min, s_max                      — normalisation constants of raw series
        metadata                          — provenance dict
    """
    processed_path = Path(paths_cfg["mg_processed_file"])
    meta_path = Path(paths_cfg["metadata_file"])

    if processed_path.exists() and not force_rebuild:
        logger.info("Loading cached dataset from %s", processed_path)
        data = np.load(processed_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        return {
            "X_train":  data["X_train"],
            "y_train":  data["y_train"],
            "X_test":   data["X_test"],
            "y_test":   data["y_test"],
            "series":   data["series"],
            "s_min":    float(metadata["s_min"]),
            "s_max":    float(metadata["s_max"]),
            "metadata": metadata,
        }

    tau        = cfg["tau"]
    window     = cfg["window_size"]
    horizon    = cfg["prediction_horizon"]
    n_samples  = cfg["total_samples"]
    train_frac = cfg.get("train_fraction", 0.8)
    do_norm    = cfg.get("normalize", True)
    washout    = cfg.get("washout", 200)

    logger.info(
        "Generating Mackey-Glass: tau=%d  n=%d  window=%d  horizon=%d  washout=%d",
        tau, n_samples, window, horizon, washout,
    )

    # ---- Generate raw series ----
    if _USE_MY_DATASET:
        # MG_series returns (X_windows, Y_scalars) with sliding window applied.
        # I just need the raw series to normalise consistently, then re-window.
        # To get the raw series: generate with window_size=1, horizon=0 trick
        # is fragile. Instead: generate at n_samples + extra and slice.
        # Cleanest: use the fallback generator for raw series, keeping
        # MG_series as an alternative path with its own windowing.
        #
        # DECISION: use MG_series directly for windowed data since it's
        # my proven implementation. Normalise X and Y jointly from raw values.
        X_raw, Y_raw = _my_MG_series(
            n_samples=n_samples,
            tau=tau,
            window_size=window,
            prediction_horizon=horizon,
            time_step=1,
            plot=False,
        )
        # X_raw: (N, window), Y_raw: (N,)
        # Normalise: fit scaler on all values seen in training windows
        all_vals = np.concatenate([X_raw.ravel(), Y_raw.ravel()])
        s_min, s_max = float(all_vals.min()), float(all_vals.max())
        rng = s_max - s_min + 1e-12

        if do_norm:
            X_norm = (X_raw - s_min) / rng
            Y_norm = (Y_raw - s_min) / rng
        else:
            X_norm, Y_norm = X_raw, Y_raw

        # Build a proxy 1-D series for the plots (first column of windows)
        series = X_norm[:, 0]
        source = "MG_series(QRC/datasets.py)"
    else:
        raw = _generate_mackey_glass_rk4(n_samples, tau, washout=washout)
        np.save(
            Path(paths_cfg["raw_dir"]) / "mackey_glass_raw.npy", raw
        )
        if do_norm:
            series, s_min, s_max = normalize_01(raw)
        else:
            series, s_min, s_max = raw, float(raw.min()), float(raw.max())

        X_norm, Y_norm = _create_windows(series, window, horizon)
        source = "RK4_fallback"

    # ---- Train/test split ----
    split  = int(len(X_norm) * train_frac)
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train = Y_norm[:split].reshape(-1, 1)
    y_test  = Y_norm[split:].reshape(-1, 1)

    metadata = {
        "tau":                tau,
        "window_size":        window,
        "prediction_horizon": horizon,
        "total_samples":      n_samples,
        "train_fraction":     train_frac,
        "washout":            washout,
        "normalize":          do_norm,
        "s_min":              s_min,
        "s_max":              s_max,
        "n_train":            int(len(X_train)),
        "n_test":             int(len(X_test)),
        "dataset_source":     source,
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    Path(paths_cfg["raw_dir"]).mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        processed_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        series=series,
    )
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Dataset saved: %d train / %d test windows  source=%s",
        len(X_train), len(X_test), source,
    )
    return {
        "X_train":  X_train,
        "y_train":  y_train,
        "X_test":   X_test,
        "y_test":   y_test,
        "series":   series,
        "s_min":    s_min,
        "s_max":    s_max,
        "metadata": metadata,
    }
