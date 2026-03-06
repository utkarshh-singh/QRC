"""
reservoir_adapter.py
--------------------
THIS IS THE ONLY FILE I NEED TO MODIFY to connect my real QRC implementation.

My QRC stack lives at QRC/ (one level above noise_study/):
    QRC/reservoirs.py   — CPRC class
    QRC/ESN.py          — ESNetwork
    QRC/circuits.py     — CPCircuit
    QRC/utils.py        — ReservoirWrapper
    QRC/datasets.py     — MG_series
    QRC/utility.py      — mapping, CPaction, MetaFibonacci

Path from this file: parents[2] = QRC/

Integration strategy
--------------------
I import CPRC and ESNetwork directly without modifying those files.
I inject the noise study's AerSimulator into CPRC by replacing its
_simulate() method at the instance level using types.MethodType.
This makes CPRC.qc_func() (called by ReservoirWrapper.compute())
run under any AerSimulator noise config transparently.

For the ideal (statevector) case I use AerSimulator(method="statevector")
instead of Qiskit's legacy Primitives Sampler, which is removed in
Qiskit >= 1.0 and would crash at runtime.

IMPORTANT: CPRC's original _simulate uses `from qiskit.primitives import Sampler`
(legacy V1 Primitives) which accesses `res.quasi_dists[0]`.  That API is
deprecated/removed in Qiskit 1.x.  My injected _simulate uses
AerSimulator.run() (backend run API) which is stable.
"""

from __future__ import annotations

import logging
import sys
from itertools import product as _iproduct
from pathlib import Path
from types import MethodType
from typing import Optional
from qiskit.compiler import transpile as _qk_transpile

import numpy as np
from sklearn.preprocessing import StandardScaler

from .noise_models import BackendBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap — add QRC/ to sys.path
# parents[0] = src/
# parents[1] = noise_study/
# parents[2] = QRC/
# ---------------------------------------------------------------------------
_QRC_ROOT = Path(__file__).resolve().parents[2]
if str(_QRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_QRC_ROOT))

# ---------------------------------------------------------------------------
# Import my QRC modules
# ---------------------------------------------------------------------------
_REAL_QRC_AVAILABLE = False
CPRC = None
ESNetwork = None

try:
    from reservoirs import CPRC as _CPRC          # noqa: E402
    from ESN import ESNetwork as _ESNetwork        # noqa: E402
    CPRC = _CPRC
    ESNetwork = _ESNetwork
    _REAL_QRC_AVAILABLE = True
    logger.info("reservoir_adapter: CPRC and ESNetwork imported from QRC/")
except ImportError as _import_err:
    logger.warning(
        "reservoir_adapter: Could not import CPRC/ESNetwork from QRC/: %s\n"
        "Falling back to dummy Ridge predictor.  "
        "Ensure QRC/ contains reservoirs.py and ESN.py.",
        _import_err,
    )


# ---------------------------------------------------------------------------
# AerSimulator injection
# ---------------------------------------------------------------------------

def _make_aer_simulate(
    simulator,
    shots: int,
    noise_model,
    transpile_flag: bool,
):
    """
    Return a bound _simulate(self, qc) replacement that runs qc on
    AerSimulator with the given noise model and returns a normalised
    probability vector over all 2^n bitstrings.

    This replaces CPRC's original _simulate which used the legacy
    qiskit.primitives.Sampler (removed in Qiskit >= 1.0).
    """
    from qiskit import transpile as _qk_transpile

    def _aer_simulate(self_cprc, qc):
        n_qubits = qc.num_qubits

        try:
            qc_run = _qk_transpile(
                qc,
                simulator,
                optimization_level=0,     # 0 = just decompose, no heavy optimisation
                basis_gates=simulator.configuration().basis_gates if hasattr(simulator, 'configuration') else None,
            )
        except Exception:
            # fallback: manual decompose loop
            qc_run = qc
            for _ in range(6):            # C-Map/P-Map are nested ~3-4 levels deep
                qc_run = qc_run.decompose()

        job    = simulator.run(qc_run, shots=shots, noise_model=noise_model)

        result = job.result()
        counts = result.get_counts()

        # Expand to full 2^n dictionary (zero-fill missing bitstrings)
        all_bits = ["".join(p) for p in _iproduct("01", repeat=n_qubits)]
        full     = {b: counts.get(b, 0) for b in all_bits}
        probs    = np.array(list(full.values()), dtype=float) / shots
        return probs

    return _aer_simulate


def _inject_backend(cprc_instance, bundle: BackendBundle) -> None:
    """
    Monkey-patch cprc_instance._simulate to use bundle.simulator.

    For ideal (statevector): inject AerSimulator with statevector method
    and shots=1 (unused — we override to return statevector probs directly).
    For all other modes: use bundle.simulator + bundle.noise_model + shots.
    """
    if bundle.noise_type == "ideal":
        # Use statevector AerSimulator: run with shots=1 then read statevector
        from qiskit import transpile as _qk_transpile
        from qiskit_aer import AerSimulator as _AerSim

        sv_sim = _AerSim(method="statevector")

        def _sv_simulate(self_cprc, qc):
            n_qubits = qc.num_qubits
            # Remove measurements for statevector, get probabilities exactly
            qc_no_meas = qc.copy()
            qc_no_meas.remove_final_measurements(inplace=True)
            from qiskit.quantum_info import Statevector
            sv = Statevector(qc_no_meas)
            probs = sv.probabilities()   # shape (2^n,), exact
            return probs

        cprc_instance._simulate = MethodType(_sv_simulate, cprc_instance)
        cprc_instance.execution_mode = "simulation"
        logger.debug("ideal: injected Statevector-based _simulate")
        return

    # All noisy modes
    shots = bundle.shots
    sim   = AerSimulator(noise_model=bundle.noise_model) \
        if bundle.noise_model is not None \
        else bundle.simulator

    fn = _make_aer_simulate(
        simulator=sim,
        shots=shots,
        noise_model=bundle.noise_model,
        transpile_flag=bundle.transpile,
    )
    cprc_instance._simulate = MethodType(fn, cprc_instance)
    cprc_instance.execution_mode = "simulation"
    logger.debug(
        "Injected AerSimulator: noise_type=%s shots=%d transpile=%s",
        bundle.noise_type, shots, bundle.transpile,
    )

# Lazy import AerSimulator here to avoid circular at module level
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_qrc_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    study_cfg: dict,
    bundle,                # BackendBundle from get_backend_bundle()
) -> dict:
    """
    Runs a single QRC experiment using ESNetwork + CPRC, faithfully
    reproducing the feedback loop from ESNetwork._apply_feedback().

    Feedback modes
    --------------
    cpk=False  (default, used here):
        modified_input[t] = α * x[t] + (1-α) * prev_quantum_state[t-1]
        Requires: len(x) == len(prev_output)  →  dim must equal window_size

    cpk=True  (kernel mode):
        x is concatenated with Z-expectation values of prev_output.
        Circuit input length = window_size + n_qubits.
        Used when kernel=True on CPRC.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from qiskit import transpile as qk_transpile

    res_cfg  = study_cfg["reservoir"]
    window_size = X_train.shape[1]

    # ------------------------------------------------------------------
    # 1.  Seed for reproducibility (Ridge + numpy ops)
    # ------------------------------------------------------------------
    seed = study_cfg.get("_run_seed", 0)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 2.  Scale inputs to [0, 2π] — circuits use these as rotation angles
    # ------------------------------------------------------------------
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler   = StandardScaler() #MinMaxScaler(feature_range=(0, np.pi))
    X_tr_sc  = scaler.fit_transform(X_train).astype(np.float32)
    X_te_sc  = scaler.transform(X_test).astype(np.float32)
    y_tr_1d  = y_train.ravel()

    # ------------------------------------------------------------------
    # 3.  Patch CPRC._simulate to use our noise-aware AerSimulator
    #     This is the key injection point — we replace the simulation
    #     backend without touching ESNetwork or CPRC source code.
    # ------------------------------------------------------------------
    from reservoirs import CPRC

    cprc = CPRC(
        dim            = window_size,
        reps           = res_cfg.get("reps", 1),
        execution_mode = "simulation",   # will be monkey-patched below
        CP_params      = res_cfg.get("cp_params", None),
        shots          = int(bundle.shots) if bundle.shots else 5000,
        optimization_level = res_cfg.get("optimization_level", 3),
        kernel         = res_cfg.get("cpk", True),
        meas_limit     = res_cfg.get("meas_limit", None),
        ETE            = res_cfg.get("ETE", False),
    )

    # Inject the noise-aware simulator by replacing _simulate on this instance
    _simulator   = bundle.simulator
    _noise_model = bundle.noise_model
    _shots       = bundle.shots
    _transpile   = bundle.transpile

    if bundle.noise_type == "ideal":
        # Use Statevector — exact, no shots
        def _patched_simulate(qc):
            from qiskit.quantum_info import Statevector
            qc_nm = qc.remove_final_measurements(inplace=False)
            # Decompose custom gates (C-Map, P-Map) before Statevector
            for _ in range(6):
                qc_nm = qc_nm.decompose()
            return Statevector(qc_nm).probabilities()

    else:
        # Use AerSimulator with the appropriate noise model
        def _patched_simulate(qc):
            try:
                qc_run = qk_transpile(qc, _simulator, optimization_level=0)
            except Exception:
                qc_run = qc
                for _ in range(8):
                    qc_run = qc_run.decompose()

            job    = _simulator.run(qc_run, shots=_shots, noise_model=_noise_model)
            result = job.result()
            counts = result.get_counts()

            n_q      = qc.num_qubits
            all_bits = ["".join(p) for p in __import__("itertools").product("01", repeat=n_q)]
            probs    = np.array([counts.get(b, 0) for b in all_bits], dtype=float) / _shots
            return probs

    # Bind the patched method to this specific instance
    import types
    cprc._simulate = types.MethodType(lambda self, qc: _patched_simulate(qc), cprc)

    # ------------------------------------------------------------------
    # 4.  Build ESNetwork
    #     CRITICAL: dim=window_size so that _apply_feedback shape matches:
    #       α * x[t]  (shape: window_size,)
    #     + (1-α) * prev_output  (shape: window_size,  ← dim=window_size)
    # ------------------------------------------------------------------
    from ESN import ESNetwork

    washout = res_cfg.get("washout", 50)

    esn = ESNetwork(
        reservoir    = cprc,
        dim          = window_size,       # ← must match x shape for feedback
        regularization = res_cfg.get("regularization", 1e-3),
        alpha        = res_cfg.get("alpha", 0.7),
        show_progress = res_cfg.get("show_progress", True),
        approach     = res_cfg.get("approach", "feedback"),
        model_type   = res_cfg.get("model_type", "ridge"),
        limit        = res_cfg.get("limit", None),
        cpk          = res_cfg.get("cpk", True),             # If kernel mode off → simple leaky feedback
        save_states  = True,
    )

    # ------------------------------------------------------------------
    # 5.  Train
    #     ESNetwork.fit() handles the full feedback loop internally:
    #       modified_input = α*x + (1-α)*prev_quantum_state
    #       quantum_state  = CPRC.qc_func(modified_input)
    #       prev_output    = quantum_state   ← fed into next step
    # ------------------------------------------------------------------
    esn.fit(X_tr_sc, y_tr_1d, washout=washout)
    states_train = esn.get_saved_states()
    # ------------------------------------------------------------------
    # 6.  Predict  (feedback loop continues through ESNetwork.predict())
    # ------------------------------------------------------------------
    esn.prev_output = np.zeros(window_size)   # reset feedback memory for test
    y_pred = esn.predict(X_te_sc)             # shape: (N_test,)

    # ------------------------------------------------------------------
    # 7.  Collect reservoir states + circuit metrics
    # ------------------------------------------------------------------
    # states_train = esn.get_saved_states()                     # (N_train-washout, state_dim)
    # Re-run predict with save flag to get test states
    # esn.saved_quantum_states = None
    # esn.save_states = True
    # esn.prev_output = np.zeros(window_size)
    # esn.predict(X_te_sc)
    states_test = None #esn.get_saved_states()                      # (N_test, state_dim)

    sample_qc = cprc.CPMap()
    sample_qc.measure_all()
    try:
        from qiskit import transpile as _t
        compiled = _t(sample_qc, _simulator, optimization_level=0) if _simulator else sample_qc.decompose()
        circuit_metrics = {
            "num_qubits": sample_qc.num_qubits,
            "depth_raw":  sample_qc.depth(),
            "depth_compiled": compiled.depth(),
            "num_cx":     compiled.count_ops().get("cx", 0),
        }
    except Exception:
        circuit_metrics = {"num_qubits": window_size, "depth_raw": -1, "depth_compiled": -1, "num_cx": -1}

    return {
        "y_test_pred":    y_pred.reshape(-1, 1),
        "y_test_true":    y_test,
        "states_train":   states_train,
        "states_test":    states_test,
        "circuit_metrics": circuit_metrics,
        "scaler":         scaler,
    }



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_test_states(esn, X_test_sc: np.ndarray) -> Optional[np.ndarray]:
    """
    Run reservoir on X_test and collect processed quantum states.
    Mirrors predict() but saves states instead of calling model.predict().
    Resets feedback to zeros first so it's independent of training state.
    """
    try:
        # Reset feedback to correct size: prev_output is set to quantum_state
        # during fit, so its size = 2^n_qubits, not dim.
        # After fit, prev_output has the right size.  We just re-initialise
        # with zeros of the same shape.
        prev_shape = np.shape(esn.prev_output)
        esn.prev_output = np.zeros(prev_shape)
        esn.show_progress = False

        states = []
        for x in X_test_sc:
            x = np.array(x, dtype=np.float32)
            modified = esn._apply_feedback(x)
            qs = esn.reservoir.compute(modified)
            out_len = int(esn.limit * len(qs)) if esn.limit else len(qs)
            states.append(qs[:out_len].flatten())

        return np.array(states)
    except Exception as exc:
        logger.warning("Could not collect test states: %s", exc)
        return None


def _circuit_metrics(cprc, dim: int, cp_params: list) -> Optional[dict]:
    """
    Build one CPMap circuit and extract depth/gate counts for provenance.
    Does NOT run the circuit — purely structural analysis.
    """
    try:
        qc = cprc.CPMap()
        # Assign all parameters to zero for structural analysis
        param_vals = {p: 0.0 for p in qc.parameters}
        bound = qc.assign_parameters(param_vals)
        return {
            "num_qubits": bound.num_qubits,
            "depth":      bound.depth(),
            "num_gates":  bound.size(),
            "cp_params":  [float(v) for v in cp_params],
            "dim":        dim,
        }
    except Exception as exc:
        logger.warning("Could not extract circuit metrics: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Dummy fallback
# ---------------------------------------------------------------------------

def _dummy_predictor(
    X_train, y_train, X_test, y_test, config, backend_bundle
) -> dict:
    """
    Ridge regression on raw input windows.
    No quantum circuit. Used when CPRC/ESNetwork are not importable.
    Results are physically meaningless but let the pipeline run end-to-end.
    """
    from sklearn.linear_model import Ridge

    y_tr = y_train.ravel()
    readout = Ridge(alpha=1e-3)
    readout.fit(X_train, y_tr)
    y_pred_1d = readout.predict(X_test)

    if backend_bundle.shots is not None:
        rng = np.random.default_rng(int(config.get("_run_seed", 0)))
        y_pred_1d = y_pred_1d + rng.normal(
            0.0, 1.0 / np.sqrt(backend_bundle.shots), size=y_pred_1d.shape
        )

    return {
        "y_test_pred":     y_pred_1d.reshape(-1, 1),
        "y_test_true":     y_test,
        "states_train":    X_train,
        "states_test":     X_test,
        "circuit_metrics": None,
        "extra_meta": {
            "adapter_mode": "dummy_fallback",
            "noise_type":   backend_bundle.noise_type,
            "shots":        backend_bundle.shots,
        },
    }
