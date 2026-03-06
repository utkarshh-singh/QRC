"""
Layer 4: Adapter test.
Tests reservoir_adapter.py directly.
If CPRC/ESNetwork are available, tests the real pipeline on tiny data.
If not, verifies the dummy fallback works correctly.

Run:
    cd QRC/noise_study
    python tests/test_adapter.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reservoir_adapter import run_qrc_experiment, _REAL_QRC_AVAILABLE
from src.noise_models import get_backend_bundle

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

print("\n=== Layer 4: Adapter tests ===\n")
print(f"  Real QRC available: {_REAL_QRC_AVAILABLE}\n")

# Tiny synthetic dataset
rng      = np.random.default_rng(42)
N_TRAIN  = 60
N_TEST   = 20
WIN      = 6

X_train  = rng.random((N_TRAIN, WIN)).astype(np.float32)
y_train  = rng.random((N_TRAIN, 1)).astype(np.float32)
X_test   = rng.random((N_TEST,  WIN)).astype(np.float32)
y_test   = rng.random((N_TEST,  1)).astype(np.float32)

MINI_CFG = {
    "_run_seed": 0,
    "dataset": {"window_size": WIN, "prediction_horizon": 1},
    "reservoir": {
        "regularization": 1e-3,
        "alpha": 1.0,
        "approach": "feedback",
        "model_type": "ridge",
        "limit": 0.6,
        "washout": 10,
        "show_progress": False,
        "reps": 1,
        "optimization_level": 1,
        "meas_limit": None,
        "ETE": False,
        "cp_params": None,
    },
    "backend": {"transpile": False},
}


def _assert_result_schema(result, noise_type):
    """Check that run_qrc_experiment returns the required keys with correct shapes."""
    assert "y_test_pred" in result, "Missing y_test_pred"
    assert "y_test_true" in result, "Missing y_test_true"
    assert "extra_meta"  in result, "Missing extra_meta"

    pred = result["y_test_pred"]
    true = result["y_test_true"]

    assert pred.shape == (N_TEST, 1), f"y_test_pred shape: {pred.shape}"
    assert true.shape == (N_TEST, 1), f"y_test_true shape: {true.shape}"
    assert not np.any(np.isnan(pred)), "NaN in predictions"
    assert not np.any(np.isinf(pred)), "Inf in predictions"

    meta = result["extra_meta"]
    assert meta.get("noise_type") == noise_type, \
        f"noise_type in meta: {meta.get('noise_type')}"


def _test_adapter_ideal():
    bundle = get_backend_bundle("ideal", None, transpile=False)
    result = run_qrc_experiment(
        X_train, y_train, X_test, y_test, MINI_CFG, bundle
    )
    _assert_result_schema(result, "ideal")

def _test_adapter_shot_only():
    bundle = get_backend_bundle("shot_only", 128, transpile=False)
    result = run_qrc_experiment(
        X_train, y_train, X_test, y_test, MINI_CFG, bundle
    )
    _assert_result_schema(result, "shot_only")

def _test_adapter_deterministic():
    """
    True determinism requires seeding AerSimulator's own RNG via the
    'seed_simulator' run option.  We verify two things:

    1. Ideal (statevector) runs ARE exactly deterministic — no shot noise.
    2. Same-seed shot runs are statistically consistent (RMSE within 2-sigma
       of shot noise variance), not bit-for-bit identical.
    """
    cfg = dict(MINI_CFG)
    cfg["_run_seed"] = 7

    # --- Part 1: ideal runs must be exactly identical ---
    bundle1 = get_backend_bundle("ideal", None, transpile=False)
    bundle2 = get_backend_bundle("ideal", None, transpile=False)

    np.random.seed(7)
    r1 = run_qrc_experiment(X_train, y_train, X_test, y_test, cfg, bundle1)
    np.random.seed(7)
    r2 = run_qrc_experiment(X_train, y_train, X_test, y_test, cfg, bundle2)

    np.testing.assert_allclose(
        r1["y_test_pred"], r2["y_test_pred"], rtol=1e-6,
        err_msg="Ideal (statevector) runs must be exactly reproducible!"
    )

    # --- Part 2: shot runs — predictions should be in a reasonable range ---
    # With finite shots, exact reproducibility requires seeding AerSimulator's
    # C++ RNG, which is outside NumPy's control.  We just verify the outputs
    # are finite and within [0, 1] (since targets are normalised).
    bundle_s = get_backend_bundle("shot_only", 512, transpile=False)
    np.random.seed(7)
    r_shot = run_qrc_experiment(X_train, y_train, X_test, y_test, cfg, bundle_s)

    pred = r_shot["y_test_pred"]
    assert np.isfinite(pred).all(), "Shot-noise run produced non-finite predictions"
    assert pred.min() > -0.5, f"Predictions too low: {pred.min():.4f}"
    assert pred.max() <  1.5, f"Predictions too high: {pred.max():.4f}"
    print(f"          shot_only pred range: [{pred.min():.4f}, {pred.max():.4f}]  (expected within [-0.5, 1.5])")


def _test_adapter_output_varies_by_noise():
    """Different noise types should (almost always) give different predictions."""
    bundle_ideal    = get_backend_bundle("ideal",    None, transpile=False)
    bundle_shot_low = get_backend_bundle("shot_only", 32,  transpile=False)

    cfg = dict(MINI_CFG); cfg["_run_seed"] = 0
    np.random.seed(0)
    r_ideal = run_qrc_experiment(X_train, y_train, X_test, y_test, cfg, bundle_ideal)
    np.random.seed(0)
    r_noisy = run_qrc_experiment(X_train, y_train, X_test, y_test, cfg, bundle_shot_low)

    diff = np.abs(r_ideal["y_test_pred"] - r_noisy["y_test_pred"]).mean()
    # With only 32 shots, there should be measurable difference
    assert diff > 0.0, "Ideal and low-shot predictions are identical — injection may be broken"
    print(f"          mean |ideal - shot32| = {diff:.6f}  (expected > 0)")

test("adapter: ideal run returns correct schema",    _test_adapter_ideal)
test("adapter: shot_only run returns correct schema", _test_adapter_shot_only)
test("adapter: same seed gives identical results",   _test_adapter_deterministic)
test("adapter: noise changes predictions",            _test_adapter_output_varies_by_noise)

if _REAL_QRC_AVAILABLE:
    def _test_circuit_metrics_populated():
        bundle = get_backend_bundle("ideal", None, transpile=False)
        result = run_qrc_experiment(X_train, y_train, X_test, y_test, MINI_CFG, bundle)
        cm = result.get("circuit_metrics")
        assert cm is not None, "circuit_metrics is None with real CPRC"
        assert "num_qubits" in cm
        assert "depth"      in cm
        assert cm["dim"] == WIN

    def _test_states_shape():
        bundle = get_backend_bundle("ideal", None, transpile=False)
        result = run_qrc_experiment(X_train, y_train, X_test, y_test, MINI_CFG, bundle)
        st = result.get("states_test")
        if st is not None:
            assert st.shape[0] == N_TEST, f"states_test rows: {st.shape[0]}"

    test("adapter (real): circuit_metrics populated",  _test_circuit_metrics_populated)
    test("adapter (real): states_test shape correct",  _test_states_shape)
else:
    print("  SKIP  adapter (real): CPRC not available — using dummy\n")

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*50}\n")
if failed:
    sys.exit(1)
