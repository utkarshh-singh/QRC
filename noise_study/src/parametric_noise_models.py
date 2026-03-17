"""
parametric_noise_models.py
--------------------------
Builds Aer NoiseModel objects from explicit error rates.
No FakeTorino — fully parametric and reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
import warnings
import logging
logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)

# 1-qubit and 2-qubit gate sets for AerSimulator
_1Q_GATES = ["u1", "u2", "u3", "rz", "sx", "x", "id", "h"]
_2Q_GATES = ["cx", "ecr", "cz"]


@dataclass
class ParametricBundle:
    noise_model:  Optional[NoiseModel]
    simulator:    AerSimulator
    shots:        int
    noise_type:   str
    noise_param:  float       # the swept parameter value
    label:        str         # human-readable, e.g. "single_qubit_p=0.01"
    transpile:    bool = True


def make_single_qubit_depol(p: float, shots: int) -> ParametricBundle:
    """Depolarizing error on all 1-qubit gates, rate p."""
    nm = NoiseModel()
    err = depolarizing_error(p, 1)
    for g in _1Q_GATES:
        nm.add_all_qubit_quantum_error(err, g)
    sim = AerSimulator(noise_model=nm)
    return ParametricBundle(nm, sim, shots, "single_qubit_depol", p,
                            f"1Q_depol_p={p:.4f}")


def make_two_qubit_depol(p: float, shots: int) -> ParametricBundle:
    """Depolarizing error on all 2-qubit gates, rate p."""
    nm = NoiseModel()
    err = depolarizing_error(p, 2)
    for g in _2Q_GATES:
        nm.add_all_qubit_quantum_error(err, g)
    sim = AerSimulator(noise_model=nm)
    return ParametricBundle(nm, sim, shots, "two_qubit_depol", p,
                            f"2Q_depol_p={p:.4f}")


def make_readout_noise(p: float, shots: int) -> ParametricBundle:
    """Symmetric readout (measurement) error, flip probability p."""
    nm = NoiseModel()
    err = ReadoutError([[1 - p, p], [p, 1 - p]])
    nm.add_all_qubit_readout_error(err)
    sim = AerSimulator(noise_model=nm)
    return ParametricBundle(nm, sim, shots, "readout", p,
                            f"readout_p={p:.4f}")


def make_relaxation(T1_us: float, T2_factor: float,
                    gate_time_ns: float, shots: int) -> ParametricBundle:
    """
    T1/T2 thermal relaxation on all 1-qubit gates.
    T2 = T1 * T2_factor (capped at 2*T1 by physics).
    gate_time in nanoseconds.
    """
    T1 = T1_us * 1e-6          # → seconds
    T2 = min(T1 * T2_factor, 2 * T1)
    gate_time = gate_time_ns * 1e-9   # → seconds

    nm = NoiseModel()
    err = thermal_relaxation_error(T1, T2, gate_time)
    for g in _1Q_GATES:
        nm.add_all_qubit_quantum_error(err, g)
    sim = AerSimulator(noise_model=nm)
    p_eff = gate_time / T1     # effective error proxy for logging
    return ParametricBundle(nm, sim, shots, "relaxation", T1_us,
                            f"relax_T1={T1_us}us")


def make_combined(p: float, shots: int,
                  T1_us: float = 100.0,
                  gate_time_ns: float = 100.0) -> ParametricBundle:
    """
    All channels simultaneously:
    - 1Q depolarizing at p
    - 2Q depolarizing at p*10 (typical ratio)
    - Readout at p
    - T1/T2 relaxation at T1=100µs
    """
    T1 = T1_us * 1e-6
    T2 = T1 * 0.5
    gate_time = gate_time_ns * 1e-9

    nm = NoiseModel()

    # 1Q depol
    err_1q = depolarizing_error(p, 1)
    for g in _1Q_GATES:
        nm.add_all_qubit_quantum_error(err_1q, g)

    # 2Q depol
    p2q = min(p * 10, 0.99)
    err_2q = depolarizing_error(p2q, 2)
    for g in _2Q_GATES:
        nm.add_all_qubit_quantum_error(err_2q, g)

    # Readout
    err_ro = ReadoutError([[1 - p, p], [p, 1 - p]])
    nm.add_all_qubit_readout_error(err_ro)

    # Relaxation (added on top)
    err_rel = thermal_relaxation_error(T1, T2, gate_time)
    for g in _1Q_GATES:
        nm.add_all_qubit_quantum_error(err_rel, g)

    sim = AerSimulator(noise_model=nm)
    return ParametricBundle(nm, sim, shots, "combined", p,
                            f"combined_p={p:.4f}")


def get_all_bundles(cfg: dict) -> list[ParametricBundle]:
    """
    Build all ParametricBundles from noise_sweep_config.yaml noise_sweep section.
    Returns flat list ordered by channel then noise level.
    """
    ns  = cfg["noise_sweep"]
    shots = cfg["shots"]
    bundles = []

    for p in ns["single_qubit_depol"]["values"]:
        bundles.append(make_single_qubit_depol(p, shots))

    for p in ns["two_qubit_depol"]["values"]:
        bundles.append(make_two_qubit_depol(p, shots))

    for p in ns["readout"]["values"]:
        bundles.append(make_readout_noise(p, shots))

    rel = ns["relaxation"]
    for T1 in rel["T1_us"]:
        bundles.append(make_relaxation(
            T1, rel["T2_factor"], rel["gate_time_ns"], shots))

    for p in ns["combined"]["values"]:
        bundles.append(make_combined(p, shots))

    return bundles   # 30 bundles total
