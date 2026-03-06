"""
metrics.py
----------
Regression metrics for time-series prediction evaluation.
All functions operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from scipy.stats import ConstantInputWarning

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute all study metrics on flattened test-set predictions.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_test, horizon) or (n_test,)
    y_pred : np.ndarray, same shape as y_true

    Returns
    -------
    dict with keys: mse, rmse, mae, r2, pearson
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()

    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))

    try:
        pearson_r, pearson_p = pearsonr(yt, yp)
        pearson = float(pearson_r)
    except Exception:
        pearson = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }
