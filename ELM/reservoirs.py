from __future__ import annotations

from itertools import product
from typing import Literal, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
from qiskit_ibm_runtime import SamplerV2 as Sampler2

from circuits import CPCircuit
from matrices import OpticalNetwork
from Permanents import RyserPermanent, ClassicalCoincidence
from utility import MetaFibonacci, mapping, CPaction

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ExecutionMode = Literal["simulation", "DM", "STT", "noise", "real_device", "fake_simulation"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def filter_top_states(prob_counts: dict, k: int) -> dict:
    """Return the k highest-probability states."""
    return dict(sorted(prob_counts.items(), key=lambda x: x[1], reverse=True)[:k])


def initialize_full_key_dict(size: int) -> dict[str, int]:
    """Return a dict keyed by every binary string of *size* bits, all zeros."""
    return {"".join(p): 0 for p in product("01", repeat=size)}


def refined_counts(result_counts: dict, size: int) -> dict[str, int]:
    """
    Merge *result_counts* into a complete dict covering all 2**size bitstrings.
    Missing bitstrings default to 0.
    """
    full = initialize_full_key_dict(size)
    full.update({k: v for k, v in result_counts.items() if k in full})
    return full


def process_counts(
    counts: dict,
    n_qubits: int,
    key_type: Literal["hexa", "decimal", "binary"] = "hexa",
) -> dict[str, int]:
    """
    Convert raw counts (hex / decimal / binary keys) to zero-padded binary keys,
    filling in any missing bitstrings with 0.
    """
    _bases = {"hexa": 16, "decimal": 10, "binary": 2}
    if key_type not in _bases:
        raise ValueError(f"key_type must be one of {list(_bases)}; got {key_type!r}")

    base = _bases[key_type]
    binary_counts = {
        format(int(k, base) if key_type != "decimal" else int(k), f"0{n_qubits}b"): v
        for k, v in counts.items()
    }
    all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
    return {b: binary_counts.get(b, 0) for b in all_bitstrings}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CPRC:
    """
    Conditional Parameterized Random Circuit kernel / feature map.

    Parameters
    ----------
    dim : int
        Number of features.
    circuit : QuantumCircuit, optional
        Pre-built circuit. If None, a CPCircuit is constructed automatically.
    reps : int
        Number of repetitions for the feature map.
    execution_mode : str
        One of 'simulation', 'DM', 'STT', 'noise', 'real_device', 'fake_simulation'.
    K_params : array-like, optional
        Fixed kernel parameters. Defaults to uniform random in [0, 1).
    CP_params : array-like, optional
        CP-gate parameters forwarded to CPCircuit.
    backend : IBM backend, optional
        Required for 'real_device' and 'fake_simulation' modes.
    shots : int
        Number of measurement shots.
    optimization_level : int
        Qiskit transpiler optimisation level (0-3).
    kernel : bool
        If True, compose circuit with its inverse for kernel evaluation.
    noise_level : float, optional
        Depolarising error probability for single-qubit gates.
        Two-qubit error is set to 10x this value.
    meas_limit : float, optional
        Fraction of qubits to measure (0, 1]. None means measure all.
    ETE : bool
        End-to-end encoding flag forwarded to CPCircuit.
    """

    def __init__(
        self,
        dim: int,
        circuit: Optional[QuantumCircuit] = None,
        reps: int = 1,
        execution_mode: ExecutionMode = "simulation",
        K_params: Optional[np.ndarray] = None,
        CP_params=None,
        backend=None,
        shots: int = 1024,
        optimization_level: int = 3,
        kernel: bool = False,
        noise_level: Optional[float] = None,
        meas_limit: Optional[float] = None,
        ETE: bool = False,
    ) -> None:
        self.dim = dim
        self.circuit = circuit
        self.reps = reps
        self.execution_mode = execution_mode
        self.backend = backend
        self.shots = shots
        self.optimization_level = optimization_level
        self.kernel = kernel
        self.noise_level = noise_level
        self.K_params = K_params if K_params is not None else np.random.uniform(0.0, 1.0, dim)
        self.CP_params = CP_params
        self.meas_limit = meas_limit
        self.ETE = ETE

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def CPMap(self) -> QuantumCircuit:
        """Build and return the parameterized CP feature-map circuit."""
        cp_circuit = CPCircuit(
            num_features=self.dim,
            reps=self.reps,
            CP_params=self.CP_params,
            ETE=self.ETE,
        )

        if self.meas_limit is None:
            return cp_circuit.CPMap()

        n_qubits = cp_circuit._num_qubits()
        meas_q = max(1, int(n_qubits * self.meas_limit))
        qc = QuantumCircuit(n_qubits, meas_q)
        # FIX: original appended to range(meas_q) instead of range(n_qubits)
        qc.append(cp_circuit.CPMap(), range(n_qubits))
        return qc

    # ------------------------------------------------------------------
    # Execution backends
    # ------------------------------------------------------------------

    def _simulate(self, qc: QuantumCircuit) -> np.ndarray:
        """Noiseless statevector simulation via Qiskit primitives."""
        sampler = Sampler()
        result = sampler.run(qc).result()
        probs = result.quasi_dists[0].values()
        return np.array(list(probs))

    def _simulate_dm(self, qc: QuantumCircuit):
        """Return the DensityMatrix of *qc*."""
        from qiskit.quantum_info import DensityMatrix
        return DensityMatrix(qc)

    def _simulate_statevector(self, qc: QuantumCircuit):
        """Return the Statevector of *qc*."""
        from qiskit.quantum_info import Statevector
        return Statevector(qc)

    def _simulate_with_noise(self, qc: QuantumCircuit) -> np.ndarray:
        """Depolarising-noise simulation via AerSimulator."""
        if self.noise_level is None:
            print("Warning: noise_level not set — falling back to noiseless simulation.")
            return self._simulate(qc)

        n_qubits = qc.num_qubits
        backend = AerSimulator()
        noise_model = self.get_depolarizing_noise_model(
            prob_1=self.noise_level,
            prob_2=self.noise_level * 10,
        )
        qc_t = transpile(qc, backend)
        result = backend.run(qc_t, shots=self.shots, noise_model=noise_model).result()
        counts = refined_counts(result.get_counts(), n_qubits)
        return np.array(list(counts.values())) / self.shots

    def _run_on_real_device(
        self, qc: QuantumCircuit, fake_simulation: bool
    ):
        """Execute on a real IBM backend or an Aer fake simulation of one."""
        n_qubits = qc.num_qubits

        if fake_simulation:
            aer_sim = AerSimulator.from_backend(self.backend)
            pm = generate_preset_pass_manager(
                backend=aer_sim, optimization_level=self.optimization_level
            )
            qc_t = pm.run(qc)
            # FIX: original had `.results` (typo) instead of `.result()`
            result = aer_sim.run([qc_t]).result()
            counts_raw = result.results[0].data.counts
            fid = process_counts(counts_raw, n_qubits, key_type="hexa").values()
            return np.array(list(fid)) / self.shots

        # Real hardware path
        pm = generate_preset_pass_manager(
            backend=self.backend, optimization_level=self.optimization_level
        )
        qc_t = pm.run(qc)
        sampler = Sampler2(mode=self.backend)
        sampler.options.default_shots = self.shots
        job = sampler.run([qc_t])
        print(f"Job submitted — ID: {job.job_id()}")
        result = job.result()
        counts = result[0].data.c.get_counts()
        fid = process_counts(counts, n_qubits, key_type="binary").values()
        return np.array(list(fid)) / self.shots, job

    def retrieve_job_result(self, job_id: str) -> np.ndarray:
        """Retrieve and process results for a previously submitted job."""
        job = self.backend.service.job(job_id)
        counts = job.result()[0].data.c.get_counts()
        fid = process_counts(counts, n_qubits=self.dim, key_type="binary").values()
        return np.array(list(fid)) / self.shots

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def qc_func(self, x: np.ndarray):
        """
        Build, bind, and execute the circuit for input *x*.

        Returns a probability vector (or DensityMatrix / Statevector for DM/STT modes).
        """
        feature_map = self.circuit if self.circuit is not None else self.CPMap()

        if self.kernel:
            circ1 = feature_map.assign_parameters(x)
            circ2 = feature_map.assign_parameters(self.K_params).inverse()
            qc = circ1.compose(circ2)
        else:
            qc = feature_map.assign_parameters(x)

        # Attach measurements
        if self.meas_limit is None:
            qc.measure_all()
        else:
            meas_q = max(1, int(qc.num_qubits * self.meas_limit))
            qc.measure(range(meas_q), range(meas_q))

        # Dispatch to execution backend
        mode = self.execution_mode
        if mode == "simulation":
            return self._simulate(qc)
        elif mode == "DM":
            qc.remove_final_measurements()
            return self._simulate_dm(qc)
        elif mode == "STT":
            qc.remove_final_measurements()
            return self._simulate_statevector(qc)
        elif mode == "noise":
            return self._simulate_with_noise(qc)
        elif mode == "real_device":
            return self._run_on_real_device(qc, fake_simulation=False)
        elif mode == "fake_simulation":
            return self._run_on_real_device(qc, fake_simulation=True)
        else:
            raise ValueError(
                f"Unknown execution_mode {mode!r}. "
                "Choose from: 'simulation', 'DM', 'STT', 'noise', 'real_device', 'fake_simulation'."
            )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_depolarizing_noise_model(prob_1: float, prob_2: float) -> noise.NoiseModel:
        """
        Build a simple depolarising noise model.

        Parameters
        ----------
        prob_1 : float
            Error probability for single-qubit gates.
        prob_2 : float
            Error probability for two-qubit gates (typically 10x prob_1).
        """
        model = noise.NoiseModel()
        model.add_all_qubit_quantum_error(noise.depolarizing_error(prob_1, 1), ["rz", "sx", "x"])
        model.add_all_qubit_quantum_error(noise.depolarizing_error(prob_2, 2), ["cx"])
        return model