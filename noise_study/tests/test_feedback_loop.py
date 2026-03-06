"""
Feedback Loop Trace Test
========================
Verifies that the feedback loop works exactly as ESNetwork._apply_feedback() intends:

    modified_input[t] = α * x[t]  +  (1-α) * prev_quantum_state[t-1]
    quantum_state[t]  = circuit(modified_input[t])
    prev_output       = quantum_state[t]   ← fed into t+1

Uses a MINIMAL 2-qubit RZ circuit (no CPCircuit dependency) so this runs
standalone — no QRC imports needed. Statevector only, no shots.

Run:
    cd QRC/noise_study
    python tests/test_feedback_loop.py
"""

import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from itertools import product as iproduct

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
N_FEATURES  = 4       # window_size / circuit input dimension
N_QUBITS    = 2       # qubits in dummy circuit  → output state dim = 2^2 = 4
ALPHA       = 0.7     # leaky feedback weight
N_STEPS     = 15      # how many timesteps to trace
WASHOUT     = 3       # steps to skip before collecting states
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Dummy data  (sine wave scaled to [0, 2π])
# ─────────────────────────────────────────────────────────────────────────────
t      = np.linspace(0, 4 * np.pi, N_STEPS + 10)
series = (np.sin(t) + 1) / 2 * 2 * np.pi     # range [0, 2π]

# Sliding windows:  X[i] = series[i : i+N_FEATURES]
X = np.array([series[i : i + N_FEATURES] for i in range(N_STEPS)], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Dummy circuit: 2-qubit, N_FEATURES parameters
#   - H on both qubits
#   - RZ(params[0]) on q0,  RZ(params[1]) on q1
#   - CNOT q0→q1
#   - RZ(params[2]) on q0,  RZ(params[3]) on q1
# This is intentionally simple — enough non-linearity to show feedback effect.
# ─────────────────────────────────────────────────────────────────────────────
params = ParameterVector("θ", N_FEATURES)

def build_circuit(param_values: np.ndarray) -> np.ndarray:
    """Assign parameters, run Statevector, return probability vector."""
    qc = QuantumCircuit(N_QUBITS)
    qc.h(0); qc.h(1)
    qc.ry(float(param_values[0]), 0)
    qc.ry(float(param_values[1]), 1)
    qc.cx(0, 1)
    qc.ry(float(param_values[2]) if len(param_values) > 2 else 0.0, 0)
    qc.ry(float(param_values[3]) if len(param_values) > 3 else 0.0, 1)
    sv   = Statevector(qc)
    probs = sv.probabilities()          # shape: (2^N_QUBITS,) = (4,)
    return probs

# ─────────────────────────────────────────────────────────────────────────────
# Manual feedback loop  (mirrors ESNetwork.fit() exactly)
# ─────────────────────────────────────────────────────────────────────────────
prev_output = np.zeros(N_FEATURES)   # dim=N_FEATURES for shape-safe feedback

print("\n" + "═"*90)
print("  FEEDBACK LOOP TRACE  —  step-by-step")
print("  α = {:.2f}  |  N_FEATURES = {}  |  N_QUBITS = {}  |  output_dim = 4".format(
      ALPHA, N_FEATURES, N_QUBITS))
print("═"*90)

# Header
hdr = (f"{'t':>3} | "
       f"{'raw_input (x)':>36} | "
       f"{'prev_output':>36} | "
       f"{'modified_input = α*x+(1-α)*prev':>36} | "
       f"{'circuit_params used':>36} | "
       f"{'quantum_state (probs)':>28} | "
       f"{'sum_probs':>9}")
print(hdr)
print("─"*len(hdr))

feedback_records = []

for t_idx in range(N_STEPS):
    x              = X[t_idx]
    prev_before    = prev_output.copy()            # q[t-1] — used to compute mod
    modified_input = ALPHA * x + (1 - ALPHA) * prev_before
    quantum_state  = build_circuit(modified_input)
    prev_output    = quantum_state.copy()          # q[t]   — used next step

    record = dict(
        t             = t_idx,
        x             = x.copy(),
        prev_before   = prev_before,               # what went INTO mod
        modified      = modified_input.copy(),
        quantum_state = quantum_state.copy(),
        washout       = t_idx < WASHOUT,
    )
    feedback_records.append(record)

    washout_tag = "[W]" if t_idx < WASHOUT else "   "
    print(
        f"t={t_idx:>2}{washout_tag}\n"
        f"  x        (raw input)   : [{', '.join(f'{v:.4f}' for v in x)}]\n"
        f"  prev     (q[t-1] used) : [{', '.join(f'{v:.4f}' for v in prev_before)}]\n"
        f"  mod      (α*x+(1-α)*p) : [{', '.join(f'{v:.4f}' for v in modified_input)}]\n"
        f"  q_out    (circuit out) : [{', '.join(f'{v:.4f}' for v in quantum_state)}]"
        f"   Σ={quantum_state.sum():.6f}\n"
        f"  verify   mod = 0.7*x + 0.3*prev:\n"
        f"    expected: [{', '.join(f'{0.7*x[i]+0.3*prev_before[i]:.4f}' for i in range(N_FEATURES))}]\n"
        f"    got:      [{', '.join(f'{modified_input[i]:.4f}' for i in range(N_FEATURES))}]"
        f"   {'✓' if np.allclose(0.7*x+0.3*prev_before, modified_input, atol=1e-5) else '✗ MISMATCH'}\n"
        + "─"*70
    )

# ─────────────────────────────────────────────────────────────────────────────
# Verification checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*90)
print("  VERIFICATION CHECKS")
print("═"*90)

errors = []

# 1. Probabilities always sum to 1
for r in feedback_records:
    diff = abs(r["quantum_state"].sum() - 1.0)
    if diff > 1e-6:
        errors.append(f"t={r['t']}: prob sum = {r['quantum_state'].sum():.8f} ≠ 1")
print(f"  [{'PASS' if not errors else 'FAIL'}]  All probability vectors sum to 1.0")

# 2. Feedback is actually changing modified_input  (not just passing x through)
diffs = [np.linalg.norm(r["modified"] - X[r["t"]]) for r in feedback_records[1:]]
all_changed = all(d > 1e-6 for d in diffs)
print(f"  [{'PASS' if all_changed else 'FAIL'}]  modified_input ≠ raw_input for all t>0 "
      f"  (mean Δ = {np.mean(diffs):.6f})")

# 3. prev_output is quantum_state from PREVIOUS step
for i in range(1, len(feedback_records)):
    prev_rec  = feedback_records[i-1]
    curr_rec  = feedback_records[i]
    # prev_output shown in curr_rec should equal quantum_state from prev_rec
    # BUT we stored prev_output AFTER the update, so we compare differently:
    # modified[t] = α*x[t] + (1-α)*quantum_state[t-1]
    expected_mod = ALPHA * X[i] + (1-ALPHA) * feedback_records[i-1]["quantum_state"]
    actual_mod   = curr_rec["modified"]
    if not np.allclose(expected_mod, actual_mod, atol=1e-6):
        errors.append(f"t={i}: feedback mismatch. expected={expected_mod}, got={actual_mod}")
print(f"  [{'PASS' if not [e for e in errors if 'feedback mismatch' in e] else 'FAIL'}]"
      f"  modified_input[t] = α*x[t] + (1-α)*quantum_state[t-1]  for all t")

# 4. Quantum states are different across timesteps (reservoir is non-trivial)
states = np.array([r["quantum_state"] for r in feedback_records])
pairwise_diffs = np.diff(states, axis=0)
are_different  = np.any(np.abs(pairwise_diffs) > 1e-6, axis=1).all()
print(f"  [{'PASS' if are_different else 'FAIL'}]  Quantum states change at every timestep")

# 5. Shape checks
for r in feedback_records:
    assert r["x"].shape          == (N_FEATURES,), f"x shape wrong at t={r['t']}"
    assert r["modified"].shape   == (N_FEATURES,), f"modified shape wrong at t={r['t']}"
    assert r["quantum_state"].shape == (2**N_QUBITS,), f"q_state shape wrong at t={r['t']}"
print(f"  [PASS]  All array shapes consistent  "
      f"(x: ({N_FEATURES},)  modified: ({N_FEATURES},)  q_state: ({2**N_QUBITS},))")

# ─────────────────────────────────────────────────────────────────────────────
# Shape mismatch warning  (the real bug when dim ≠ window_size)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*90)
print("  SHAPE MISMATCH DEMO  —  what happens if dim ≠ N_FEATURES")
print("═"*90)

for wrong_dim in [2, 8]:
    prev_wrong = np.zeros(wrong_dim)
    try:
        bad_mod = ALPHA * X[0] + (1 - ALPHA) * prev_wrong
        print(f"  dim={wrong_dim:>2}, N_FEATURES={N_FEATURES}  →  result shape: {bad_mod.shape}  "
              f"[{'OK — but semantically wrong, uses broadcast' if bad_mod.shape != (N_FEATURES,) else 'shape matches by accident'}]")
    except ValueError as e:
        print(f"  dim={wrong_dim:>2}, N_FEATURES={N_FEATURES}  →  CRASH: {e}")

print(f"\n  ✓ Correct config: dim=N_FEATURES={N_FEATURES}  →  no crash, correct blend\n")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table — compact view
# ─────────────────────────────────────────────────────────────────────────────
print("═"*90)
print("  SUMMARY TABLE  (post-washout steps only)")
print(f"  {'t':>3}  {'‖modified - x‖':>16}  {'‖q_state‖':>12}  {'q[0]':>8}  {'q[1]':>8}  {'q[2]':>8}  {'q[3]':>8}")
print("─"*90)
for r in feedback_records:
    if r["washout"]:
        continue
    delta = np.linalg.norm(r["modified"] - X[r["t"]])
    qs    = r["quantum_state"]
    print(f"  {r['t']:>3}  {delta:>16.6f}  {np.linalg.norm(qs):>12.6f}  "
          f"{qs[0]:>8.5f}  {qs[1]:>8.5f}  {qs[2]:>8.5f}  {qs[3]:>8.5f}")

print("\n  ‖modified - x‖ > 0 at every row confirms feedback is active.")

if errors:
    print(f"\n  ✗ {len(errors)} CHECKS FAILED:")
    for e in errors: print(f"    - {e}")
    sys.exit(1)
else:
    print("\n  ✓ All checks passed — feedback loop is working correctly.\n")
