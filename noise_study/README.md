# QRC Noise Study

A reproducible, lightweight experiment suite for analysing the effect of
quantum hardware noise on a **Quantum Reservoir Computing (QRC)** model
trained on the **Mackey–Glass** time-series prediction task.

---

## Purpose

This package isolates and quantifies how different noise channels degrade
QRC performance.  It is designed as a plug-in experiment layer around an
**existing QRC implementation** (located elsewhere in the project).  Only
one file — `src/reservoir_adapter.py` — needs to be edited to connect the
real reservoir.

---

## Experiment Design

| Group | Description | Runs |
|---|---|---|
| **Ideal** | Noiseless statevector simulation | 1 config × 3 seeds = 3 |
| **Shot sweep** | Shot-only noise, varying shots | 4 configs × 3 seeds = 12 |
| **Noise ablation** | 6 noise types at 1024 shots | 6 configs × 3 seeds = 18 |
| **Total** | | **33 runs** |

### Noise types

| Label | Description | Source |
|---|---|---|
| `ideal` | Exact statevector, no noise | AerSimulator[statevector] |
| `shot_only` | Finite shots, no device noise | AerSimulator[qasm] |
| `readout_only` | Readout (measurement) error only | FakeTorino calibration |
| `single_qubit_only` | 1-qubit gate errors only | FakeTorino noise model |
| `two_qubit_only` | 2-qubit gate errors only | FakeTorino noise model |
| `relaxation_only` | T1/T2 thermal relaxation only | FakeTorino properties |
| `full_backend` | Full realistic noise model | NoiseModel.from_backend(FakeTorino) |

### Dataset

- **Mackey–Glass** delay differential equation, τ=17
- 10 000 samples generated, washout 200, 80/20 train/test split
- Sliding window: input=20 steps, predict=20 steps ahead
- Min-max normalised to [0, 1]

---

## Folder Structure

```
noise_study/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py              ← orchestrate full study
├── run_one.py              ← run a single experiment
├── aggregate_results.py    ← collect results into CSVs
├── make_plots.py           ← generate figures
├── configs/
│   ├── study_config.yaml   ← all study parameters
│   └── paths.yaml          ← filesystem layout
├── src/
│   ├── data.py             ← MG generator + windowing
│   ├── noise_models.py     ← BackendBundle factory functions
│   ├── reservoir_adapter.py← *** EDIT THIS FILE ***
│   ├── runner.py           ← single-run executor
│   ├── metrics.py          ← MSE, RMSE, MAE, R², Pearson
│   ├── io_utils.py         ← save/load utilities + manifest
│   └── plot_utils.py       ← figures
├── manifests/
│   └── planned_runs.csv    ← auto-generated run table
├── data/raw/               ← raw MG series (.npy)
├── data/processed/         ← windowed dataset (.npz) + metadata
├── runs/<run_id>/          ← per-run artifacts
├── results/aggregated/     ← master_results.csv, grouped_results.csv
├── results/figures/        ← .pdf figures
├── results/tables/         ← (manual use; e.g. LaTeX tables)
└── logs/                   ← study.log
```

---

## How to Plug In My Real QRC Code

Open `src/reservoir_adapter.py` and locate the section:

```python
# === MY QRC INTEGRATION POINT ===
```

Replace the dummy Ridge predictor with calls to my actual reservoir.
The pseudocode template inside the file shows the expected pattern:

```python
from my_qrc_module import QuantumReservoir

reservoir = QuantumReservoir(
    n_qubits    = config["reservoir"]["n_qubits"],
    backend     = backend_bundle.simulator,
    noise_model = backend_bundle.noise_model,
    shots       = backend_bundle.shots,
    transpile   = backend_bundle.transpile,
)

states_train = reservoir.transform(X_train)
states_test  = reservoir.transform(X_test)

readout = Ridge(alpha=1e-4)
readout.fit(states_train, y_train)
y_test_pred = readout.predict(states_test)

return {
    "y_test_pred"    : y_test_pred,
    "y_test_true"    : y_test,
    "states_train"   : states_train,
    "states_test"    : states_test,
    "circuit_metrics": reservoir.get_circuit_metrics(),
}
```

That is the **only change** needed.  All saving, logging, metrics, and
plotting happens automatically.

---

## How to Run

### 0. Install dependencies

```bash
cd QRC/noise_study
pip install -r requirements.txt
```

### 1. Generate manifest only (dry run)

```bash
python run_all.py --dry-run
```

### 2. Run a single experiment (for testing)

```bash
# By noise type
python run_one.py --noise-type ideal --shots none --seed 0
python run_one.py --noise-type shot_only --shots 1024 --seed 1
python run_one.py --noise-type full_backend --shots 1024 --seed 0

# By run_id from manifest
python run_one.py --run-id ablation_full_backend_s1024_seed0
```

### 3. Run the full study

```bash
python run_all.py
# Already-successful runs are automatically skipped on re-run.
# To force fresh dataset generation:
python run_all.py --force-rebuild-data
```

### 4. Aggregate results

```bash
python aggregate_results.py
# Produces:
#   results/aggregated/master_results.csv
#   results/aggregated/grouped_results.csv
```

### 5. Generate plots

```bash
python make_plots.py
# Produces PDFs in results/figures/
# To specify which run to use for the prediction trace:
python make_plots.py --trace-run-id ideal_seed0
```

---

## Per-Run Outputs

Each run writes to `runs/<run_id>/`:

| File | Description |
|---|---|
| `config_used.yaml` | Full config snapshot |
| `metadata.json` | run_id, seed, noise_type, shots, backend, versions, timing |
| `metrics.json` | MSE, RMSE, MAE, R², Pearson |
| `predictions.csv` | y_true, y_pred columns |
| `run.log` | Detailed log for this run |
| `status.json` | `{"status": "success"}` or `{"status": "failed"}` |
| `error.txt` | Full traceback (failure only) |
| `circuit_metrics.json` | Circuit depth/gate counts (if adapter returns them) |
| `reservoir_states_train.npy` | Reservoir state matrix (if adapter returns it) |
| `reservoir_states_test.npy` | Reservoir state matrix (if adapter returns it) |

---

## Limitations

1. **Noise channel isolation is approximate.**  
   Qiskit Aer's `NoiseModel.from_backend()` bundles depolarising +
   thermal relaxation into each gate error object.  The isolated modes
   (`single_qubit_only`, `two_qubit_only`, `relaxation_only`) are
   constructed by filtering or rebuilding errors, not by accessing
   separate physical channels.  This is standard practice but not a
   perfect decomposition.

2. **Dummy adapter produces physically meaningless results.**  
   Until `reservoir_adapter.py` is wired to the real QRC pipeline, all
   metrics reflect a simple linear model, not quantum reservoir dynamics.

3. **FakeTorino noise model is static calibration data.**  
   It does not reflect live device drift or crosstalk beyond what Qiskit
   reports in the fake backend properties.

4. **No circuit transpilation in dummy mode.**  
   The dummy adapter does not build or submit any quantum circuits, so
   `transpile=True` has no effect until the real adapter is connected.

5. **seeds = [0, 1, 2] is minimal for statistical estimates.**  
   Three seeds give a rough variance estimate.  For publication, increase
   to at least 5–10 seeds in `study_config.yaml`.








cd QRC/noise_study

# Step 1 — dry run: generate + print the manifest without executing anything
python run_all.py --dry-run

# Step 2 — smoke test: run just one experiment end-to-end first
python run_one.py --noise-type ideal --shots none --seed 0

# Step 3 — check the output
cat runs/ideal_seed0/metrics.json
cat runs/ideal_seed0/status.json    # should say {"status": "success"}

# Step 4 — run all 33 experiments (already-successful runs are skipped on re-run)
python run_all.py

# Step 5 — collect results into CSVs
python aggregate_results.py

# Step 6 — generate plots
python make_plots.py
