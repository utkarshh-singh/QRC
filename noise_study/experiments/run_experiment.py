"""
QRC Experiment Runner
=====================
Runs a single experiment by ID from config.json.

Usage:
    cd QRC/noise_study
    python experiments/run_experiment.py --id exp_01
    python experiments/run_experiment.py --id exp_04
    python experiments/run_experiment.py --list        # show all experiment IDs
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--id",     type=str, help="Experiment ID, e.g. exp_01")
parser.add_argument("--list",   action="store_true", help="List all experiment IDs and exit")
parser.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "config.json"),
                    help="Path to config.json")
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────
with open(args.config) as f:
    full_config = json.load(f)

shared = full_config["shared"]
experiments = {e["id"]: e for e in full_config["experiments"]}

if args.list:
    print("\nAvailable experiments:")
    for eid, ecfg in experiments.items():
        print(f"  {eid:10s}  {ecfg['name']:35s}  {ecfg['description']}")
    print()
    sys.exit(0)

if not args.id:
    print("ERROR: provide --id or --list"); sys.exit(1)
if args.id not in experiments:
    print(f"ERROR: '{args.id}' not found. Use --list to see valid IDs."); sys.exit(1)

# Merge shared + experiment-specific (experiment keys override shared)
exp = experiments[args.id]
cfg = {**shared, **exp}   # exp keys win over shared

os.makedirs("results", exist_ok=True)
np.random.seed(cfg["seed"])

# ── 1. Generate Mackey-Glass data ─────────────────────────────
from datasets import MG_series

print(f"\n{'═'*65}")
print(f"  Experiment : {cfg['id']}  —  {cfg['name']}")
print(f"  {cfg['description']}")
print(f"{'═'*65}")
print(f"  Generating MG_series: n_samples={cfg['n_samples']}  "
      f"tau={cfg['tau']}  window={cfg['window_size']}  "
      f"horizon={cfg['prediction_horizon']}  time_step={cfg['time_step']}")

X_all, y_all = MG_series(
    n_samples          = cfg["n_samples"],
    b                  = cfg["b"],
    c                  = cfg["c"],
    tau                = cfg["tau"],
    window_size        = cfg["window_size"],
    prediction_horizon = cfg["prediction_horizon"],
    time_step          = cfg["time_step"],
)

n_total = len(X_all)
n_train = int(n_total * cfg["train_ratio"])
n_test  = n_total - n_train

X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
y_train,      y_test     = y_all[:n_train], y_all[n_train:]

print(f"  Total samples: {n_total}  →  train: {n_train}  test: {n_test}")

# Scale X to [0, π] for circuit rotation angles
scaler  = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)

W = cfg["window_size"]

# ── 2. Build CPRC ──────────────────────────────────────────────
from reservoirs import CPRC

cprc = CPRC(
    dim            = W,
    reps           = cfg["reps"],
    execution_mode = cfg["execution_mode"],
    shots          = cfg["shots"],
    kernel         = cfg["cpk"],        # explicit — don't rely on ESNetwork patch alone
    noise_level    = cfg["noise_level"],
    CP_params      = cfg["CP_params"],
    meas_limit     = cfg["meas_limit"],
    ETE            = cfg["ETE"],
)

# ── 3. Build ESNetwork ─────────────────────────────────────────
from ESN import ESNetwork
from circuits import CPCircuit

n_qubits  = CPCircuit(num_features=W, reps=cfg["reps"])._num_qubits()
state_dim = 2 ** n_qubits

esn = ESNetwork(
    reservoir      = cprc,
    dim            = W,
    alpha          = cfg["alpha"],
    regularization = cfg["regularization"],
    approach       = "feedback",
    cpk            = cfg["cpk"],
    model_type     = cfg["model_type"],
    show_progress  = True,
    save_states    = True,
)

print(f"  n_qubits: {n_qubits}   state_dim: {state_dim}   "
      f"cpk: {cfg['cpk']}   alpha: {cfg['alpha']}")
print(f"  washout: {cfg['washout']}   noise_level: {cfg['noise_level']}\n")

# ── 4. Train ───────────────────────────────────────────────────
esn.fit(X_train, y_train, washout=cfg["washout"])
states_train = esn.get_saved_states()
print(f"\n  Training states shape : {states_train.shape}")

# ── 5. Predict ─────────────────────────────────────────────────
esn.prev_output = np.zeros(W)    # reset feedback memory before test
y_pred = esn.predict(X_test)

# ── 6. Metrics ─────────────────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
nmse = mse / np.var(y_test)     # <1.0 = better than mean predictor

print(f"\n{'═'*65}")
print(f"  RESULTS")
print(f"{'═'*65}")
print(f"  RMSE : {rmse:.6f}")
print(f"  NMSE : {nmse:.6f}   (< 1.0 = better than mean predictor)")
print(f"  R²   : {r2:.6f}   (1.0 = perfect)")
print(f"{'═'*65}\n")

# ── 7. Save results ────────────────────────────────────────────
result = {
    "id":          cfg["id"],
    "name":        cfg["name"],
    "description": cfg["description"],
    "config":      cfg,
    "data": {
        "n_total": n_total,
        "n_train": n_train,
        "n_test":  n_test,
    },
    "circuit": {
        "n_qubits":  n_qubits,
        "state_dim": state_dim,
    },
    "states_train_shape": list(states_train.shape),
    "metrics": {
        "rmse": float(rmse),
        "nmse": float(nmse),
        "r2":   float(r2),
        "mse":  float(mse),
    },
}

json_path = f"results/{cfg['id']}_{cfg['name']}.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2)

np.save(f"results/{cfg['id']}_{cfg['name']}_y_pred.npy",        y_pred)
np.save(f"results/{cfg['id']}_{cfg['name']}_y_test.npy",        y_test)
np.save(f"results/{cfg['id']}_{cfg['name']}_states_train.npy",  states_train)

print(f"  Saved → results/{cfg['id']}_{cfg['name']}.json\n")
