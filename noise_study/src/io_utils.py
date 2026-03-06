"""
io_utils.py
-----------
Saving, loading, and manifest management utilities.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML / JSON helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def save_json(data: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable.")


# ---------------------------------------------------------------------------
# Version capture
# ---------------------------------------------------------------------------

def capture_versions() -> dict:
    packages = [
        "qiskit", "qiskit-aer", "qiskit-ibm-runtime",
        "numpy", "scipy", "pandas", "scikit-learn",
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------

def make_run_dir(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    config_used: dict,
    metadata: dict,
    metrics: dict,
    y_test_pred: np.ndarray,
    y_test_true: np.ndarray,
    states_train: Optional[np.ndarray] = None,
    states_test: Optional[np.ndarray] = None,
    circuit_metrics: Optional[dict] = None,
) -> None:
    save_yaml(config_used, run_dir / "config_used.yaml")
    save_json(metadata,    run_dir / "metadata.json")
    save_json(metrics,     run_dir / "metrics.json")

    pd.DataFrame({
        "y_true": y_test_true.ravel(),
        "y_pred": y_test_pred.ravel(),
    }).to_csv(run_dir / "predictions.csv", index=False)

    if states_train is not None:
        np.save(run_dir / "reservoir_states_train.npy", states_train)
    if states_test is not None:
        np.save(run_dir / "reservoir_states_test.npy", states_test)
    if circuit_metrics is not None:
        save_json(circuit_metrics, run_dir / "circuit_metrics.json")


def save_error_traceback(run_dir: Path, exc: Exception) -> None:
    tb = traceback.format_exc()
    (run_dir / "error.txt").write_text(tb, encoding="utf-8")
    save_json({"status": "failed", "error": str(exc)}, run_dir / "status.json")
    logger.error("Run %s FAILED: %s", run_dir.name, exc)


def mark_run_success(run_dir: Path) -> None:
    save_json({"status": "success"}, run_dir / "status.json")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = ["run_id", "experiment_type", "noise_type", "shots", "seed"]


def _make_run_id(
    exp_type: str,
    shots: Optional[int],
    seed: int,
    noise_type: Optional[str] = None,
) -> str:
    """
    Build a deterministic, human-readable run_id.

    Examples
    --------
    ideal_seed0
    shot_sweep_s256_seed1
    ablation_full_backend_s1024_seed2
    manual_shot_only_s1024_seed0
    """
    parts = [exp_type]
    # Always include noise_type for ablation AND manual runs so run_ids
    # are unique when multiple noise types share the same shots+seed.
    if noise_type and exp_type in ("ablation", "manual"):
        parts.append(noise_type)
    if shots is not None:
        parts.append(f"s{shots}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def generate_manifest(study_cfg: dict) -> pd.DataFrame:
    """Build planned_runs DataFrame from study_config."""
    rows  = []
    seeds = study_cfg["seeds"]

    # Ideal
    for seed in seeds:
        rows.append({
            "run_id":          _make_run_id("ideal", None, seed),
            "experiment_type": "ideal",
            "noise_type":      "ideal",
            "shots":           None,
            "seed":            seed,
        })

    # Shot sweep
    for shots in study_cfg["shot_sweep"]["shots"]:
        for seed in seeds:
            rows.append({
                "run_id":          _make_run_id("shot_sweep", shots, seed),
                "experiment_type": "shot_sweep",
                "noise_type":      "shot_only",
                "shots":           shots,
                "seed":            seed,
            })

    # Noise ablation
    abl_shots = study_cfg["ablation"]["shots"]
    for noise_type in study_cfg["ablation"]["noise_types"]:
        for seed in seeds:
            rows.append({
                "run_id":          _make_run_id("ablation", abl_shots, seed, noise_type),
                "experiment_type": "ablation",
                "noise_type":      noise_type,
                "shots":           abl_shots,
                "seed":            seed,
            })

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Manifest saved: %d runs → %s", len(df), path)


def load_manifest(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(runs_root: Path) -> pd.DataFrame:
    """Collect metrics + metadata from all successful runs into one DataFrame."""
    records = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        status_file = run_dir / "status.json"
        if not status_file.exists():
            continue
        try:
            status = load_json(status_file)
        except Exception:
            continue
        if status.get("status") != "success":
            continue
        try:
            meta = load_json(run_dir / "metadata.json")
            mets = load_json(run_dir / "metrics.json")
            records.append({**meta, **mets})
        except Exception as exc:
            logger.warning("Could not load run %s: %s", run_dir.name, exc)

    if not records:
        logger.warning("No successful runs found in %s", runs_root)
        return pd.DataFrame()

    return pd.DataFrame(records)
