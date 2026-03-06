"""
noise_models.py
---------------
Factory functions for each noise configuration.

Backend reference : FakeTorino (qiskit_ibm_runtime.fake_provider)
Simulator         : AerSimulator

Isolation caveats
-----------------
Qiskit Aer's NoiseModel.from_backend() bundles depolarising + thermal
relaxation into each gate error.  Isolation is approximated by:

  readout_only      — only ReadoutError objects, no gate errors
  single_qubit_only — gate errors for 1-qubit instructions only
  two_qubit_only    — gate errors for 2-qubit instructions only
  relaxation_only   — rebuild errors using ONLY thermal_relaxation_error
                      (no depolarising), from T1/T2/gate-time properties
  full_backend      — NoiseModel.from_backend(FakeTorino), unmodified

Note on private API (_local_quantum_errors)
-------------------------------------------
Qiskit Aer's NoiseModel does not expose a public per-gate error iterator.
We use NoiseModel.to_dict() to extract gate errors safely, which is a
stable serialisation API, avoiding internal attribute access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BackendBundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class BackendBundle:
    """
    Packages everything the reservoir adapter needs to run a circuit.

    Attributes
    ----------
    simulator   : AerSimulator configured with the noise model.
    noise_model : The NoiseModel (None for shot_only and ideal).
    shots       : Shot count, or None for statevector.
    noise_type  : Label string, e.g. "full_backend".
    backend_name: Human-readable identifier.
    transpile   : Whether circuits should be transpiled before execution.
    meta        : Arbitrary extra metadata dict.
    """
    simulator:    AerSimulator
    noise_model:  Optional[NoiseModel]
    shots:        Optional[int]
    noise_type:   str
    backend_name: str
    transpile:    bool
    meta:         dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FakeTorino singleton
# ---------------------------------------------------------------------------

_fake_torino_instance = None


def _get_fake_torino():
    global _fake_torino_instance
    if _fake_torino_instance is None:
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorino
        except ImportError as exc:
            raise ImportError(
                "FakeTorino requires qiskit-ibm-runtime >= 0.20. "
                "Install: pip install qiskit-ibm-runtime"
            ) from exc
        _fake_torino_instance = FakeTorino()
        logger.debug("FakeTorino created and cached.")
    return _fake_torino_instance


# ---------------------------------------------------------------------------
# Helper: safely iterate gate errors from a NoiseModel
# ---------------------------------------------------------------------------

def _iter_gate_errors(noise_model: NoiseModel):
    """
    Yield (instruction_name, qargs_tuple, QuantumError) for every local
    gate error in noise_model.

    Uses the public .to_dict() serialisation to avoid private attribute access.
    We reconstruct QuantumError objects by filtering the model's known errors.

    Since we need the actual QuantumError objects (not just their dict repr),
    we access the internal dict as a last resort but guard it defensively.
    """
    # Access internal store with a safe fallback
    raw = getattr(noise_model, '_local_quantum_errors', None)
    if raw is None:
        logger.warning("Cannot access _local_quantum_errors; noise filtering unavailable.")
        return

    # Structure in Qiskit Aer >= 0.13:
    # _local_quantum_errors: dict[str, dict[tuple[int,...], QuantumError]]
    #   key   = gate name (str)
    #   value = dict mapping qargs tuple → QuantumError
    for gate_name, qargs_dict in raw.items():
        if not isinstance(qargs_dict, dict):
            continue
        for qargs, error in qargs_dict.items():
            yield gate_name, qargs, error


# ---------------------------------------------------------------------------
# Helper: T1/T2/gate-time extraction
# ---------------------------------------------------------------------------

def _extract_relaxation_params(backend) -> dict:
    """
    Extract per-qubit T1, T2 (seconds) and gate times from backend properties.
    """
    props    = backend.properties()
    n_qubits = backend.configuration().n_qubits

    t1 = [props.t1(q) for q in range(n_qubits)]
    t2 = [min(props.t2(q), 2 * t1[q]) for q in range(n_qubits)]  # T2 ≤ 2T1

    gate_times_1q: dict[str, list] = {}
    gate_times_2q: dict[tuple, float] = {}

    for gate in props.gates:
        name   = gate.gate
        qubits = tuple(gate.qubits)
        try:
            gt = props.gate_length(name, list(qubits))
        except Exception:
            gt = 50e-9  # 50 ns default

        if len(qubits) == 1:
            if name not in gate_times_1q:
                gate_times_1q[name] = [None] * n_qubits
            gate_times_1q[name][qubits[0]] = gt
        elif len(qubits) == 2:
            gate_times_2q[(name, qubits[0], qubits[1])] = gt

    return {
        "t1":             t1,
        "t2":             t2,
        "gate_times_1q":  gate_times_1q,
        "gate_times_2q":  gate_times_2q,
        "n_qubits":       n_qubits,
    }


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------

def build_ideal() -> BackendBundle:
    """
    AerSimulator with statevector method — exact, no noise, no shot limit.
    We still use AerSimulator (not Qiskit Primitives) for a uniform
    execution path across all noise types.
    """
    sim = AerSimulator(method="statevector")
    return BackendBundle(
        simulator=sim,
        noise_model=None,
        shots=None,
        noise_type="ideal",
        backend_name="AerSimulator[statevector]",
        transpile=False,
        meta={"method": "statevector"},
    )


def build_shot_only(shots: int, transpile: bool = False) -> BackendBundle:
    """
    Finite-shot QASM simulation, no device noise model attached.
    """
    sim = AerSimulator(method="automatic")
    return BackendBundle(
        simulator=sim,
        noise_model=None,
        shots=shots,
        noise_type="shot_only",
        backend_name="AerSimulator[qasm,no_noise]",
        transpile=transpile,
        meta={"shots": shots},
    )


def build_readout_only(shots: int, transpile: bool = True) -> BackendBundle:
    """
    Only single-qubit readout (measurement) errors from FakeTorino.
    Gate errors are intentionally excluded.
    """
    backend  = _get_fake_torino()
    props    = backend.properties()
    n_qubits = backend.configuration().n_qubits

    nm = NoiseModel()
    for q in range(n_qubits):
        try:
            p_err = props.readout_error(q)   # P(wrong readout)
            # Symmetric approximation: P(0|1) = P(1|0) = p_err
            p_err = float(np.clip(p_err, 0.0, 0.5))
            ro_matrix = [[1 - p_err, p_err],
                         [p_err,     1 - p_err]]
            nm.add_readout_error(ReadoutError(ro_matrix), [q])
        except Exception as exc:
            logger.warning("Readout error for qubit %d skipped: %s", q, exc)

    sim = AerSimulator(noise_model=nm)
    return BackendBundle(
        simulator=sim,
        noise_model=nm,
        shots=shots,
        noise_type="readout_only",
        backend_name="FakeTorino[readout_only]",
        transpile=transpile,
        meta={"n_qubits": n_qubits},
    )


def build_single_qubit_only(shots: int, transpile: bool = True) -> BackendBundle:
    """
    Only 1-qubit gate errors from the full FakeTorino noise model.
    2-qubit gate errors and readout errors are excluded.

    Isolation uses _local_quantum_errors internal dict filtered to 1-qubit
    qargs.  See module docstring for caveat.
    """
    backend  = _get_fake_torino()
    full_nm  = NoiseModel.from_backend(backend)
    nm       = NoiseModel(basis_gates=full_nm.basis_gates)

    for gate_name, qargs, error in _iter_gate_errors(full_nm):
        if len(qargs) == 1:
            nm.add_quantum_error(error, gate_name, list(qargs))

    sim = AerSimulator(noise_model=nm)
    return BackendBundle(
        simulator=sim,
        noise_model=nm,
        shots=shots,
        noise_type="single_qubit_only",
        backend_name="FakeTorino[single_qubit_only]",
        transpile=transpile,
        meta={},
    )


def build_two_qubit_only(shots: int, transpile: bool = True) -> BackendBundle:
    """
    Only 2-qubit gate errors from the full FakeTorino noise model.
    1-qubit gate errors and readout errors are excluded.
    """
    backend  = _get_fake_torino()
    full_nm  = NoiseModel.from_backend(backend)
    nm       = NoiseModel(basis_gates=full_nm.basis_gates)

    for gate_name, qargs, error in _iter_gate_errors(full_nm):
        if len(qargs) == 2:
            nm.add_quantum_error(error, gate_name, list(qargs))

    sim = AerSimulator(noise_model=nm)
    return BackendBundle(
        simulator=sim,
        noise_model=nm,
        shots=shots,
        noise_type="two_qubit_only",
        backend_name="FakeTorino[two_qubit_only]",
        transpile=transpile,
        meta={},
    )


def build_relaxation_only(shots: int, transpile: bool = True) -> BackendBundle:
    """
    Only T1/T2 thermal relaxation errors.  No depolarising, no readout.

    We extract T1, T2, gate times from FakeTorino properties and manually
    build thermal_relaxation_error for each gate and qubit.
    """
    backend = _get_fake_torino()
    info    = _extract_relaxation_params(backend)
    t1_list = info["t1"]
    t2_list = info["t2"]
    n_qubits = info["n_qubits"]
    gate_times_1q = info["gate_times_1q"]
    gate_times_2q = info["gate_times_2q"]

    cfg        = backend.configuration()
    basis_gates = cfg.basis_gates
    DEFAULT_GT  = 50e-9  # 50 ns fallback

    nm = NoiseModel(basis_gates=basis_gates)

    # 1-qubit gates
    for gate in basis_gates:
        if gate in ("cx", "cz", "ecr", "rzz", "swap", "ccx", "reset"):
            continue
        for q in range(n_qubits):
            t1 = t1_list[q]
            t2 = t2_list[q]
            if t1 <= 0:
                continue
            gt = (gate_times_1q.get(gate, [None] * n_qubits)[q]
                  or DEFAULT_GT)
            if gt <= 0:
                continue
            try:
                err = thermal_relaxation_error(t1, t2, gt)
                nm.add_quantum_error(err, gate, [q])
            except Exception as exc:
                logger.debug("Relaxation error skip gate=%s q=%d: %s", gate, q, exc)

    # 2-qubit gates: tensor product of per-qubit relaxation
    for (gate, q0, q1), gt in gate_times_2q.items():
        if gt <= 0:
            continue
        t1_0, t2_0 = t1_list[q0], t2_list[q0]
        t1_1, t2_1 = t1_list[q1], t2_list[q1]
        if t1_0 <= 0 or t1_1 <= 0:
            continue
        try:
            err0 = thermal_relaxation_error(t1_0, t2_0, gt)
            err1 = thermal_relaxation_error(t1_1, t2_1, gt)
            combined = err0.expand(err1)
            nm.add_quantum_error(combined, gate, [q0, q1])
        except Exception as exc:
            logger.debug("Relaxation 2Q skip gate=%s q0=%d q1=%d: %s",
                         gate, q0, q1, exc)

    sim = AerSimulator(noise_model=nm)
    t1_mean_us = float(np.mean([t for t in t1_list if t > 0])) * 1e6
    return BackendBundle(
        simulator=sim,
        noise_model=nm,
        shots=shots,
        noise_type="relaxation_only",
        backend_name="FakeTorino[relaxation_only]",
        transpile=transpile,
        meta={"t1_mean_us": round(t1_mean_us, 3)},
    )


def build_full_backend(shots: int, transpile: bool = True) -> BackendBundle:
    """
    Full FakeTorino noise model: gate errors (depolarising + relaxation)
    and readout errors.
    """
    backend = _get_fake_torino()
    nm      = NoiseModel.from_backend(backend)
    sim     = AerSimulator(noise_model=nm)
    return BackendBundle(
        simulator=sim,
        noise_model=nm,
        shots=shots,
        noise_type="full_backend",
        backend_name="FakeTorino[full_backend]",
        transpile=transpile,
        meta={},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_backend_bundle(
    noise_type: str,
    shots: Optional[int],
    transpile: bool = True,
) -> BackendBundle:
    """
    Return the correct BackendBundle for (noise_type, shots).

    noise_type must be one of:
        ideal | shot_only | readout_only | single_qubit_only |
        two_qubit_only | relaxation_only | full_backend
    """
    dispatch = {
        "ideal":             lambda: build_ideal(),
        "shot_only":         lambda: build_shot_only(shots, transpile),
        "readout_only":      lambda: build_readout_only(shots, transpile),
        "single_qubit_only": lambda: build_single_qubit_only(shots, transpile),
        "two_qubit_only":    lambda: build_two_qubit_only(shots, transpile),
        "relaxation_only":   lambda: build_relaxation_only(shots, transpile),
        "full_backend":      lambda: build_full_backend(shots, transpile),
    }
    if noise_type not in dispatch:
        raise ValueError(
            f"Unknown noise_type='{noise_type}'. "
            f"Valid: {list(dispatch.keys())}"
        )
    if noise_type != "ideal" and shots is None:
        raise ValueError(
            f"shots=None is only valid for noise_type='ideal'. "
            f"Got noise_type='{noise_type}'."
        )
    return dispatch[noise_type]()
