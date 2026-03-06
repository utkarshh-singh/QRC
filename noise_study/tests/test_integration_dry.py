"""
Layer 3: Dry-run integration test.
Runs the full pipeline end-to-end using the dummy adapter (no CPRC needed).
Uses tiny data (n_samples=200) and seeds=[0] for speed.

Run:
    cd QRC/noise_study
    python tests/test_integration_dry.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import generate_manifest, save_manifest, load_yaml, load_json
from src.data import build_dataset
from src.runner import execute_run

ROOT = Path(__file__).resolve().parents[1]

passed = 0
failed = 0

def ok(name): global passed; passed += 1; print(f"  PASS  {name}")
def fail(name, e): global failed; failed += 1; print(f"  FAIL  {name}: {e}")

def test(name, fn):
    try:
        fn()
        ok(name)
    except Exception as e:
        import traceback
        fail(name, e)
        traceback.print_exc()

print("\n=== Layer 3: Dry-run integration test ===\n")
print("  Uses dummy adapter — no quantum circuits executed.\n")

# Minimal study config for fast testing
MINI_CFG = {
    "dataset": {
        "name": "mackey_glass",
        "tau": 17,
        "window_size": 10,
        "prediction_horizon": 5,
        "total_samples": 500,
        "train_fraction": 0.8,
        "washout": 50,
        "normalize": True,
    },
    "seeds": [0],
    "shot_sweep": {"shots": [64, 256], "noise_type": "shot_only"},
    "ablation": {
        "shots": 64,
        "noise_types": ["shot_only", "full_backend"],
    },
    "backend": {"name": "FakeTorino", "simulator": "AerSimulator", "transpile": False},
    "reservoir": {
        "cp_params": None,
        "reps": 1,
        "regularization": 1e-3,
        "alpha": 1.0,
        "approach": "feedback",
        "model_type": "ridge",
        "limit": 0.6,
        "washout": 50,
        "show_progress": False,
        "optimization_level": 1,
        "meas_limit": None,
        "ETE": False,
    },
    "metrics": ["mse", "rmse", "mae", "r2", "pearson"],
}

PATHS_CFG = None   # built dynamically per test using tmpdir


def _make_paths_cfg(tmpdir: Path) -> dict:
    return {
        "runs_dir": str(tmpdir / "runs"),
        "data": {
            "raw_dir":             str(tmpdir / "data/raw"),
            "processed_dir":       str(tmpdir / "data/processed"),
            "mg_processed_file":   str(tmpdir / "data/processed/mg_dataset.npz"),
            "metadata_file":       str(tmpdir / "data/processed/metadata.json"),
        },
        "manifests": {
            "planned_runs": str(tmpdir / "manifests/planned_runs.csv"),
        },
        "results": {
            "aggregated_dir": str(tmpdir / "results/aggregated"),
            "figures_dir":    str(tmpdir / "results/figures"),
            "tables_dir":     str(tmpdir / "results/tables"),
            "master_csv":     str(tmpdir / "results/aggregated/master.csv"),
            "grouped_csv":    str(tmpdir / "results/aggregated/grouped.csv"),
        },
        "logs": {"study_log": str(tmpdir / "logs/study.log")},
    }


def _test_dataset_build():
    with tempfile.TemporaryDirectory() as td:
        paths = _make_paths_cfg(Path(td))
        ds = build_dataset(MINI_CFG["dataset"], paths["data"])
        assert ds["X_train"].shape[1] == 10,  "window_size mismatch"
        assert ds["y_train"].shape[1] == 1,   "y_train should be (N,1)"
        assert len(ds["X_train"]) > 100,      "too few training samples"
        assert len(ds["X_test"])  > 20,       "too few test samples"
        assert ds["X_train"].min() >= 0.0,    "not normalised (min)"
        assert ds["X_train"].max() <= 1.0,    "not normalised (max)"
        assert not np.any(np.isnan(ds["X_train"])), "NaN in X_train"

test("dataset: build_dataset produces correct shapes", _test_dataset_build)


def _test_manifest_generation():
    df = generate_manifest(MINI_CFG)
    assert df["run_id"].nunique() == len(df), "Duplicate run_ids"
    assert set(df.columns) == {
        "run_id", "experiment_type", "noise_type", "shots", "seed"
    }
    # With seeds=[0], shots=[64,256], ablation=[shot_only,full_backend]:
    # ideal=1, shot_sweep=2, ablation=2 → total=5
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"

test("manifest: correct rows for mini config", _test_manifest_generation)


def _test_single_ideal_run():
    with tempfile.TemporaryDirectory() as td:
        paths    = _make_paths_cfg(Path(td))
        runs_root = Path(paths["runs_dir"])
        dataset  = build_dataset(MINI_CFG["dataset"], paths["data"])

        row = pd.Series({
            "run_id":          "ideal_seed0",
            "experiment_type": "ideal",
            "noise_type":      "ideal",
            "shots":           float("nan"),
            "seed":            0,
        })

        cfg = dict(MINI_CFG)
        success = execute_run(
            row=row,
            study_cfg=cfg,
            paths_cfg=paths,
            runs_root=runs_root,
            dataset=dataset,
        )
        assert success, "ideal run returned False"

        run_dir = runs_root / "ideal_seed0"
        for fname in ["config_used.yaml", "metadata.json",
                      "metrics.json", "predictions.csv", "status.json"]:
            assert (run_dir / fname).exists(), f"Missing: {fname}"

        status = load_json(run_dir / "status.json")
        assert status["status"] == "success"

        metrics = load_json(run_dir / "metrics.json")
        for k in ("mse", "rmse", "mae", "r2", "pearson"):
            assert k in metrics, f"Missing metric: {k}"
            assert np.isfinite(metrics[k]), f"Non-finite metric {k}={metrics[k]}"

test("integration: ideal run creates all artifacts", _test_single_ideal_run)


def _test_single_noisy_run():
    with tempfile.TemporaryDirectory() as td:
        paths     = _make_paths_cfg(Path(td))
        runs_root = Path(paths["runs_dir"])
        dataset   = build_dataset(MINI_CFG["dataset"], paths["data"])

        row = pd.Series({
            "run_id":          "shot_sweep_s64_seed0",
            "experiment_type": "shot_sweep",
            "noise_type":      "shot_only",
            "shots":           64,
            "seed":            0,
        })

        success = execute_run(
            row=row,
            study_cfg=MINI_CFG,
            paths_cfg=paths,
            runs_root=runs_root,
            dataset=dataset,
        )
        assert success, "shot_only run returned False"

        run_dir = runs_root / "shot_sweep_s64_seed0"
        status  = load_json(run_dir / "status.json")
        assert status["status"] == "success"

test("integration: shot_only run completes successfully", _test_single_noisy_run)


def _test_failed_run_is_handled():
    """Verify that a broken run saves traceback and returns False without crash."""
    with tempfile.TemporaryDirectory() as td:
        paths     = _make_paths_cfg(Path(td))
        runs_root = Path(paths["runs_dir"])
        dataset   = build_dataset(MINI_CFG["dataset"], paths["data"])

        row = pd.Series({
            "run_id":          "bad_run",
            "experiment_type": "ablation",
            "noise_type":      "nonexistent_noise_type",   # will raise ValueError
            "shots":           64,
            "seed":            0,
        })

        success = execute_run(
            row=row,
            study_cfg=MINI_CFG,
            paths_cfg=paths,
            runs_root=runs_root,
            dataset=dataset,
        )
        assert not success, "Should have returned False for bad run"

        run_dir    = runs_root / "bad_run"
        status     = load_json(run_dir / "status.json")
        error_file = run_dir / "error.txt"
        assert status["status"] == "failed"
        assert error_file.exists(), "error.txt should be saved"
        assert "ValueError" in error_file.read_text()

test("integration: failed run saves traceback and returns False",
     _test_failed_run_is_handled)


def _test_aggregation():
    """Run two runs then aggregate into master CSV."""
    from src.io_utils import aggregate_runs
    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        paths     = _make_paths_cfg(Path(td))
        runs_root = Path(paths["runs_dir"])
        dataset   = build_dataset(MINI_CFG["dataset"], paths["data"])

        for noise_type, shots, run_id in [
            ("ideal",     float("nan"), "ideal_seed0"),
            ("shot_only", 64,           "shot_sweep_s64_seed0"),
        ]:
            row = pd.Series({
                "run_id":          run_id,
                "experiment_type": "ideal" if noise_type == "ideal" else "shot_sweep",
                "noise_type":      noise_type,
                "shots":           shots,
                "seed":            0,
            })
            execute_run(row=row, study_cfg=MINI_CFG, paths_cfg=paths,
                        runs_root=runs_root, dataset=dataset)

        master = aggregate_runs(runs_root)
        assert len(master) == 2, f"Expected 2 rows, got {len(master)}"
        assert "rmse" in master.columns
        assert "run_id" in master.columns
        assert master["rmse"].notna().all(), "NaN in RMSE"

test("integration: aggregation produces correct master CSV", _test_aggregation)


print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*50}\n")
if failed:
    sys.exit(1)
