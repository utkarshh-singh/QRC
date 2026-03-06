"""
Layer 2 (continued): Noise model unit tests.
Tests that each BackendBundle has the correct structure and that the
AerSimulator can actually execute a trivial circuit under each config.

Run:
    cd QRC/noise_study
    python tests/test_noise_models.py
"""

import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise_models import get_backend_bundle

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
        import traceback
        fail(name, e)
        traceback.print_exc()


def _make_trivial_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """A simple Bell-state circuit for execution tests."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def _check_bundle_structure(bundle, expected_noise_type, expected_shots):
    """Assert that bundle fields are consistent."""
    assert bundle.noise_type == expected_noise_type, \
        f"noise_type: got {bundle.noise_type}"
    assert bundle.shots == expected_shots, \
        f"shots: got {bundle.shots}"
    assert bundle.simulator is not None, "simulator is None"
    assert isinstance(bundle.backend_name, str) and len(bundle.backend_name) > 0


def _run_circuit_on_bundle(bundle, n_qubits: int = 2) -> np.ndarray:
    """
    Execute a trivial circuit on bundle.simulator and return a probability vector.
    Returns a numpy array; for ideal uses statevector probabilities.
    """
    qc = _make_trivial_circuit(n_qubits)

    if bundle.noise_type == "ideal":
        # Ideal path: use Statevector (no shots)
        from qiskit.quantum_info import Statevector
        qc_no_meas = qc.copy()
        qc_no_meas.remove_final_measurements(inplace=True)
        sv = Statevector(qc_no_meas)
        return sv.probabilities()
    else:
        qc_run = transpile(qc, bundle.simulator) if bundle.transpile else qc
        job    = bundle.simulator.run(
            qc_run,
            shots=bundle.shots,
            noise_model=bundle.noise_model,
        )
        counts = job.result().get_counts()
        all_bits = [f"{i:0{n_qubits}b}" for i in range(2 ** n_qubits)]
        full = {b: counts.get(b, 0) for b in all_bits}
        return np.array(list(full.values()), dtype=float) / bundle.shots


print("\n=== Layer 2: Noise model tests ===\n")

# Structure tests (fast — no circuit execution)
def _test_ideal_structure():
    b = get_backend_bundle("ideal", None, transpile=False)
    _check_bundle_structure(b, "ideal", None)
    assert b.noise_model is None
    assert b.transpile is False

def _test_shot_only_structure():
    b = get_backend_bundle("shot_only", 512, transpile=False)
    _check_bundle_structure(b, "shot_only", 512)
    assert b.noise_model is None

def _test_full_backend_structure():
    b = get_backend_bundle("full_backend", 1024, transpile=True)
    _check_bundle_structure(b, "full_backend", 1024)
    assert b.noise_model is not None
    assert b.transpile is True

def _test_invalid_noise_type():
    try:
        get_backend_bundle("totally_wrong", 1024)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def _test_shots_none_raises():
    try:
        get_backend_bundle("shot_only", None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

test("noise_models: ideal structure",        _test_ideal_structure)
test("noise_models: shot_only structure",    _test_shot_only_structure)
test("noise_models: full_backend structure", _test_full_backend_structure)
test("noise_models: invalid type raises",    _test_invalid_noise_type)
test("noise_models: shots=None raises",      _test_shots_none_raises)

# Execution tests (slightly slower — actually runs circuits)
print("\n  --- Circuit execution tests (requires Qiskit Aer) ---\n")

def _test_ideal_execution():
    b = get_backend_bundle("ideal", None, transpile=False)
    probs = _run_circuit_on_bundle(b, n_qubits=2)
    assert len(probs) == 4
    assert abs(probs.sum() - 1.0) < 1e-6
    # Bell state: only |00> and |11> should have ~0.5 each
    assert probs[0] > 0.4  # |00>
    assert probs[3] > 0.4  # |11>

def _test_shot_only_execution():
    b = get_backend_bundle("shot_only", 2048, transpile=False)
    probs = _run_circuit_on_bundle(b, n_qubits=2)
    assert len(probs) == 4
    assert abs(probs.sum() - 1.0) < 1e-6
    # With finite shots, Bell state still mostly |00> and |11>
    assert probs[0] > 0.3
    assert probs[3] > 0.3

def _test_full_backend_execution():
    b = get_backend_bundle("full_backend", 1024, transpile=True)
    probs = _run_circuit_on_bundle(b, n_qubits=2)
    assert len(probs) == 4
    assert abs(probs.sum() - 1.0) < 1e-3  # noisy — allow small rounding

def _test_readout_only_execution():
    b = get_backend_bundle("readout_only", 1024, transpile=True)
    probs = _run_circuit_on_bundle(b, n_qubits=2)
    assert abs(probs.sum() - 1.0) < 1e-3

def _test_relaxation_only_execution():
    b = get_backend_bundle("relaxation_only", 1024, transpile=True)
    probs = _run_circuit_on_bundle(b, n_qubits=2)
    assert abs(probs.sum() - 1.0) < 1e-3

test("execution: ideal Bell state",        _test_ideal_execution)
test("execution: shot_only Bell state",    _test_shot_only_execution)
test("execution: full_backend Bell state", _test_full_backend_execution)
test("execution: readout_only Bell state", _test_readout_only_execution)
test("execution: relaxation_only Bell state", _test_relaxation_only_execution)

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*50}\n")
if failed:
    sys.exit(1)
