"""
runner.py
---------
Executes a single experiment run from a manifest row.
Tolerates individual failures — saves traceback and continues.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import build_dataset
from .io_utils import (
    capture_versions,
    load_yaml,
    make_run_dir,
    mark_run_success,
    save_error_traceback,
    save_run_artifacts,
)
from .metrics import compute_metrics
from .noise_models import get_backend_bundle
from .reservoir_adapter import run_qrc_experiment

logger = logging.getLogger(__name__)

# Track per-run FileHandlers so we can remove them cleanly
_run_file_handlers: list[logging.FileHandler] = []


def execute_run(
    row: pd.Series,
    study_cfg: dict,
    paths_cfg: dict,
    runs_root: Path,
    dataset: Optional[dict] = None,
) -> bool:
    """
    Execute one experiment run.

    Parameters
    ----------
    row        : one row of planned_runs.csv
    study_cfg  : full study_config.yaml
    paths_cfg  : full paths.yaml
    runs_root  : Path to runs/ directory
    dataset    : pre-loaded dataset dict (avoids reload per run)

    Returns
    -------
    bool : True = success, False = failure
    """
    run_id     = str(row["run_id"])
    noise_type = str(row["noise_type"])
    shots      = None if pd.isna(row["shots"]) else int(row["shots"])
    seed       = int(row["seed"])

    run_dir = make_run_dir(runs_root, run_id)
    _attach_run_log(run_dir)

    logger.info("=" * 60)
    logger.info("START  run_id=%s  noise=%s  shots=%s  seed=%d",
                run_id, noise_type, shots, seed)
    t_start = time.perf_counter()

    try:
        np.random.seed(seed)

        if dataset is None:
            dataset = build_dataset(study_cfg["dataset"], paths_cfg["data"])

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_test  = dataset["X_test"]
        y_test  = dataset["y_test"]

        transpile_flag = study_cfg["backend"].get("transpile", True)
        bundle = get_backend_bundle(
            noise_type=noise_type,
            shots=shots,
            transpile=transpile_flag,
        )

        run_cfg = dict(study_cfg)
        run_cfg["_run_seed"] = seed

        result = run_qrc_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            study_cfg=run_cfg,
            bundle=bundle,
        )

        y_pred = result["y_test_pred"]   # (n_test, 1)
        y_true = result["y_test_true"]   # (n_test, 1)

        metrics = compute_metrics(y_true, y_pred)
        elapsed = time.perf_counter() - t_start

        metadata = {
            "run_id":                   run_id,
            "timestamp":                datetime.now(timezone.utc).isoformat(),
            "seed":                     seed,
            "experiment_type":          str(row["experiment_type"]),
            "noise_type":               noise_type,
            "shots":                    shots,
            "backend_name":             bundle.backend_name,
            "transpile":                bundle.transpile,
            "elapsed_seconds":          round(elapsed, 3),
            "dataset_tau":              study_cfg["dataset"]["tau"],
            "dataset_window_size":      study_cfg["dataset"]["window_size"],
            "dataset_prediction_horizon": study_cfg["dataset"]["prediction_horizon"],
            "dataset_total_samples":    study_cfg["dataset"]["total_samples"],
            "n_train":                  int(len(X_train)),
            "n_test":                   int(len(X_test)),
            "package_versions":         capture_versions(),
        }

        if result.get("circuit_metrics"):
            metadata["circuit_metrics_summary"] = result["circuit_metrics"]
        if result.get("extra_meta"):
            metadata["adapter_extra"] = result["extra_meta"]

        save_run_artifacts(
            run_dir=run_dir,
            config_used=study_cfg,
            metadata=metadata,
            metrics=metrics,
            y_test_pred=y_pred,
            y_test_true=y_true,
            states_train=result.get("states_train"),
            states_test=result.get("states_test"),
            circuit_metrics=result.get("circuit_metrics"),
        )
        mark_run_success(run_dir)

        logger.info(
            "DONE   run_id=%s  RMSE=%.6f  R2=%.4f  elapsed=%.1fs",
            run_id, metrics["rmse"], metrics["r2"], elapsed,
        )
        return True

    except Exception as exc:
        save_error_traceback(run_dir, exc)
        return False

    finally:
        _detach_run_log()


# ---------------------------------------------------------------------------
# Per-run logging helpers
# ---------------------------------------------------------------------------

def _attach_run_log(run_dir: Path) -> None:
    """Add a FileHandler pointing at run_dir/run.log to the root logger."""
    _detach_run_log()   # remove any leftover handler from a previous run

    log_path = run_dir / "run.log"
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    _run_file_handlers.append(fh)


def _detach_run_log() -> None:
    """Remove and close all per-run FileHandlers."""
    root = logging.getLogger()
    for fh in list(_run_file_handlers):
        root.removeHandler(fh)
        fh.close()
        _run_file_handlers.remove(fh)
