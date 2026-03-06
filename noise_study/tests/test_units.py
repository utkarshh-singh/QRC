"""
Layer 2: Unit tests — one function per module.
No quantum circuits, no CPRC needed.

Run:
    cd QRC/noise_study
    python tests/test_units.py
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_metrics
from src.io_utils import (
    _make_run_id, generate_manifest, save_json, load_json,
    save_yaml, load_yaml, save_run_artifacts, mark_run_success,
    save_error_traceback,
)

passed = 0
failed = 0

def ok(name):
    global passed
    passed += 1
    print(f"  PASS  {name}")

def fail(name, e):
    global failed
    failed += 1
    print(f"  FAIL  {name}: {e}")

def test(name, fn):
    try:
        fn()
        ok(name)
    except Exception as e:
        fail(name, e)


print("\n=== Layer 2: Unit tests ===\n")

# ---- metrics.py ----

def _test_metrics_perfect():
    y = np.linspace(0, 1, 100).reshape(-1, 1)
    m = compute_metrics(y, y)
    assert m["mse"]     == 0.0,  f"MSE={m['mse']}"
    assert m["rmse"]    == 0.0,  f"RMSE={m['rmse']}"
    assert m["mae"]     == 0.0,  f"MAE={m['mae']}"
    assert abs(m["r2"] - 1.0) < 1e-10, f"R2={m['r2']}"
    assert abs(m["pearson"] - 1.0) < 1e-10, f"Pearson={m['pearson']}"

def _test_metrics_random():
    rng = np.random.default_rng(0)
    y_true = rng.random((50, 1))
    y_pred = y_true + rng.normal(0, 0.05, (50, 1))
    m = compute_metrics(y_true, y_pred)
    assert 0 < m["rmse"] < 0.5
    assert 0.9 < m["r2"] < 1.0
    assert 0.9 < m["pearson"] < 1.0

def _test_metrics_keys():
    y = np.ones((10, 1))
    yp = np.ones((10, 1)) * 1.1
    m = compute_metrics(y, yp)
    for k in ("mse", "rmse", "mae", "r2", "pearson"):
        assert k in m, f"Missing key: {k}"

test("metrics: perfect prediction", _test_metrics_perfect)
test("metrics: noisy prediction",   _test_metrics_random)
test("metrics: all keys present",   _test_metrics_keys)

# ---- io_utils.py — run_id generation ----

def _test_run_id_ideal():
    rid = _make_run_id("ideal", None, 0)
    assert rid == "ideal_seed0", f"Got: {rid}"

def _test_run_id_shot_sweep():
    rid = _make_run_id("shot_sweep", 1024, 2)
    assert rid == "shot_sweep_s1024_seed2", f"Got: {rid}"

def _test_run_id_ablation():
    rid = _make_run_id("ablation", 1024, 1, "full_backend")
    assert rid == "ablation_full_backend_s1024_seed1", f"Got: {rid}"

def _test_run_id_manual():
    rid = _make_run_id("manual", 512, 0, "readout_only")
    assert rid == "manual_readout_only_s512_seed0", f"Got: {rid}"

def _test_run_id_uniqueness():
    """All run_ids in a standard manifest must be unique."""
    cfg = {
        "seeds": [0, 1, 2],
        "shot_sweep": {"shots": [64, 256, 1024, 4096]},
        "ablation": {
            "shots": 1024,
            "noise_types": ["shot_only", "readout_only", "single_qubit_only",
                            "two_qubit_only", "relaxation_only", "full_backend"],
        },
    }
    df = generate_manifest(cfg)
    assert df["run_id"].nunique() == len(df), (
        f"Duplicate run_ids found!\n{df[df.duplicated('run_id')]['run_id'].tolist()}"
    )

test("run_id: ideal",      _test_run_id_ideal)
test("run_id: shot_sweep", _test_run_id_shot_sweep)
test("run_id: ablation",   _test_run_id_ablation)
test("run_id: manual",     _test_run_id_manual)
test("run_id: uniqueness across manifest", _test_run_id_uniqueness)

# ---- io_utils.py — manifest shape ----

def _test_manifest_count():
    cfg = {
        "seeds": [0, 1, 2],
        "shot_sweep": {"shots": [64, 256, 1024, 4096]},
        "ablation": {
            "shots": 1024,
            "noise_types": ["shot_only", "readout_only", "single_qubit_only",
                            "two_qubit_only", "relaxation_only", "full_backend"],
        },
    }
    df = generate_manifest(cfg)
    n_ideal    = 3               # 1 config × 3 seeds
    n_shot     = 4 * 3           # 4 shot values × 3 seeds
    n_ablation = 6 * 3           # 6 noise types × 3 seeds
    expected   = n_ideal + n_shot + n_ablation   # = 33
    assert len(df) == expected, f"Expected {expected} rows, got {len(df)}"
    assert set(df.columns) == {"run_id", "experiment_type",
                                "noise_type", "shots", "seed"}

test("manifest: correct row count (33)", _test_manifest_count)

# ---- io_utils.py — JSON/YAML round-trip ----

def _test_json_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.json"
        data = {"a": 1, "b": [1, 2, 3], "c": np.float32(3.14)}
        save_json(data, p)
        loaded = load_json(p)
        assert loaded["a"] == 1
        assert loaded["b"] == [1, 2, 3]
        assert abs(loaded["c"] - 3.14) < 1e-4

def _test_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.yaml"
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        save_yaml(data, p)
        loaded = load_yaml(p)
        assert loaded == data

test("io: JSON round-trip with numpy scalar", _test_json_roundtrip)
test("io: YAML round-trip",                  _test_yaml_roundtrip)

# ---- io_utils.py — artifact saving ----

def _test_save_run_artifacts():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "test_run"
        run_dir.mkdir()
        y_true = np.random.rand(50, 1)
        y_pred = y_true + 0.01
        save_run_artifacts(
            run_dir=run_dir,
            config_used={"test": True},
            metadata={"run_id": "test_run", "seed": 0},
            metrics={"mse": 0.01, "rmse": 0.1, "mae": 0.08, "r2": 0.99, "pearson": 0.999},
            y_test_pred=y_pred,
            y_test_true=y_true,
            states_train=np.random.rand(40, 8),
            states_test=np.random.rand(10, 8),
            circuit_metrics={"depth": 12, "num_qubits": 4},
        )
        mark_run_success(run_dir)

        assert (run_dir / "config_used.yaml").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.csv").exists()
        assert (run_dir / "reservoir_states_train.npy").exists()
        assert (run_dir / "reservoir_states_test.npy").exists()
        assert (run_dir / "circuit_metrics.json").exists()
        assert (run_dir / "status.json").exists()

        status = load_json(run_dir / "status.json")
        assert status["status"] == "success"

        import pandas as pd
        preds = pd.read_csv(run_dir / "predictions.csv")
        assert list(preds.columns) == ["y_true", "y_pred"]
        assert len(preds) == 50

def _test_save_error_traceback():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "failed_run"
        run_dir.mkdir()
        try:
            raise ValueError("test error for audit")
        except Exception as exc:
            save_error_traceback(run_dir, exc)
        assert (run_dir / "error.txt").exists()
        assert (run_dir / "status.json").exists()
        status = load_json(run_dir / "status.json")
        assert status["status"] == "failed"
        assert "test error" in status["error"]

test("io: save_run_artifacts creates all files",  _test_save_run_artifacts)
test("io: save_error_traceback creates status",   _test_save_error_traceback)

# ---- data.py — fallback RK4 generator ----

def _test_rk4_generator():
    from src.data import _generate_mackey_glass_rk4
    series = _generate_mackey_glass_rk4(n_samples=500, tau=17, washout=100)
    assert series.shape == (500,), f"Shape: {series.shape}"
    assert not np.any(np.isnan(series)), "NaN in series"
    assert not np.any(np.isinf(series)), "Inf in series"
    # MG series with tau=17 should oscillate roughly in [0.4, 1.4]
    assert series.min() > 0.3, f"Min too low: {series.min():.4f}"
    assert series.max() < 1.8, f"Max too high: {series.max():.4f}"

def _test_windowing():
    from src.data import _create_windows
    series = np.arange(100, dtype=float)
    X, Y = _create_windows(series, window_size=5, prediction_horizon=3)
    # First window: [0,1,2,3,4], target: series[5+3-1] = series[7] = 7
    assert X[0].tolist() == [0, 1, 2, 3, 4], f"X[0]={X[0]}"
    assert Y[0] == 7.0, f"Y[0]={Y[0]}"
    # Shape check
    expected_n = 100 - 5 - 3 + 1
    assert len(X) == expected_n, f"len(X)={len(X)}, expected {expected_n}"

def _test_normalize():
    from src.data import normalize_01
    arr = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    normed, s_min, s_max = normalize_01(arr)
    assert abs(normed.min()) < 1e-8, f"Min not 0: {normed.min()}"
    assert abs(normed.max() - 1.0) < 1e-6, f"Max not 1: {normed.max()}"
    assert s_min == 0.0
    assert s_max == 8.0

test("data: RK4 generator produces valid series",  _test_rk4_generator)
test("data: windowing indices are correct",         _test_windowing)
test("data: normalize_01 bounds are [0,1]",         _test_normalize)

# ---- Summary ----
print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*50}\n")
if failed:
    sys.exit(1)
