"""
Test: ESNetwork with cpk=True
Verifies the full pipeline runs without errors and feedback is active.

Run:
    cd QRC/noise_study
    python tests/test_cpk_true.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from reservoirs import CPRC
from ESN import ESNetwork

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 6
REPS        = 1
ALPHA       = 0.7
WASHOUT     = 3
N_SAMPLES   = 20
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# Dummy sine-wave data scaled to [0, π]
# ─────────────────────────────────────────────────────────────────
t      = np.linspace(0, 4 * np.pi, N_SAMPLES + WINDOW_SIZE + 1)
series = (np.sin(t) + 1) / 2 * np.pi
X      = np.array([series[i:i+WINDOW_SIZE] for i in range(N_SAMPLES)], dtype=np.float32)
y      = series[WINDOW_SIZE : N_SAMPLES + WINDOW_SIZE].astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# Build CPRC + ESNetwork (cpk=True)
# ESNetwork will automatically set reservoir.kernel = True
# _apply_feedback will:
#   1. extract Z-expectations from prev_output  → shape (n_qubits,)
#   2. zero-pad to shape (dim,) = (WINDOW_SIZE,)
#   3. concatenate with x                       → shape (2*WINDOW_SIZE,)
#   4. qc_func splits in half → val1, val2 → kernel circuit U(val1)·U†(val2)
# ─────────────────────────────────────────────────────────────────
cprc = CPRC(
    dim            = WINDOW_SIZE,
    reps           = REPS,
    execution_mode = 'simulation',
    shots          = 1024,
    kernel         = True
)

esn = ESNetwork(
    reservoir      = cprc,
    dim            = WINDOW_SIZE,   # prev_output_k = zeros(WINDOW_SIZE)
    alpha          = ALPHA,
    regularization = 1e-3,
    approach       = 'feedback',
    cpk            = True,          # ← kernel mode ON
    model_type     = 'ridge',
    show_progress  = True,
    save_states    = True,
)

init_prev_shape = esn.prev_output.shape 

print(f"\n{'='*60}")
print(f"  Config: window_size={WINDOW_SIZE}  reps={REPS}  alpha={ALPHA}  cpk=True")
print(f"  X shape: {X.shape}   y shape: {y.shape}")
print(f"  reservoir.kernel set by ESNetwork: {esn.reservoir.kernel}")
print(f"  prev_output init shape: {esn.prev_output.shape}  (should be ({WINDOW_SIZE},))")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────
# Manually trace first 3 steps to verify feedback
# ─────────────────────────────────────────────────────────────────
from ESN import extract_expectation_values

print("── Manual trace of _apply_feedback (first 3 steps) ──")
prev = np.zeros(WINDOW_SIZE)

for t_idx in range(3):
    x = X[t_idx]

    # Replicate _apply_feedback cpk=True logic exactly
    prev_out      = extract_expectation_values(prev)           # (2^n,) → (n_qubits,)
    prev_output_k = np.zeros(WINDOW_SIZE)
    prev_output_k[:len(prev_out)] = prev_out * ALPHA
    mod           = np.concatenate((x, prev_output_k))        # (2*WINDOW_SIZE,)

    print(f"\nt={t_idx}")
    print(f"  x              : {np.round(x, 4)}")
    print(f"  prev (zeros/q) : {np.round(prev, 4)}")
    print(f"  z_expectations : {np.round(prev_out, 4)}  (len={len(prev_out)})")
    print(f"  prev_output_k  : {np.round(prev_output_k, 4)}  (len={len(prev_output_k)})")
    print(f"  mod = concat   : {np.round(mod, 4)}  (len={len(mod)})  ← goes to circuit")
    print(f"  val1 ([:half]) : {np.round(mod[:WINDOW_SIZE], 4)}  ← forward pass U(val1)")
    print(f"  val2 ([half:]) : {np.round(mod[WINDOW_SIZE:], 4)}  ← inverse pass U†(val2)")

    # Run actual circuit to get q_out for next step
    q_out = cprc.qc_func(mod)
    prev  = q_out
    print(f"  q_out          : {np.round(q_out, 4)}  Σ={q_out.sum():.6f}")

# ─────────────────────────────────────────────────────────────────
# Full fit
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Running esn.fit() ...")
print(f"{'='*60}")

esn.fit(X, y, washout=WASHOUT)
states = esn.get_saved_states()

print(f"\n  Saved states shape : {states.shape}")
print(f"  Expected           : ({N_SAMPLES - WASHOUT}, {states.shape[1]})")

# ─────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────
esn.prev_output = np.zeros(WINDOW_SIZE)   # reset memory for test
y_pred = esn.predict(X)
print(f"  y_pred shape       : {y_pred.shape}")

# ─────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  CHECKS")
print(f"{'='*60}")

checks = {
    "reservoir.kernel == True":
        esn.reservoir.kernel == True,

    f"prev_output init shape == ({WINDOW_SIZE},)":
        init_prev_shape == (WINDOW_SIZE,),

    f"saved states shape == ({N_SAMPLES-WASHOUT}, {states.shape[1]})":
        states.shape[0] == N_SAMPLES - WASHOUT,

    "states are NOT all identical (feedback is active)":
        not np.allclose(states[0], states[1], atol=1e-6),

    "y_pred has correct length":
        len(y_pred) == N_SAMPLES,

    "no NaN or Inf in states":
        not np.any(np.isnan(states)) and not np.any(np.isinf(states)),
}

all_pass = True
for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  [{status}]  {name}")

print(f"\n  {'✓ All checks passed — cpk=True pipeline working correctly.' if all_pass else '✗ Some checks failed.'}\n")
if not all_pass:
    sys.exit(1)
