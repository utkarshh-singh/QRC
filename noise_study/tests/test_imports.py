"""
Layer 1: Import smoke test.
Verifies every src/ module is importable and exposes its public API.

Run:
    cd QRC/noise_study
    python tests/test_imports.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  OK   {label}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        errors.append((label, e))

print("\n=== Layer 1: Import smoke tests ===\n")

check("src.data imports",
      lambda: __import__("src.data", fromlist=["build_dataset"]))

check("src.noise_models imports",
      lambda: __import__("src.noise_models", fromlist=["get_backend_bundle", "BackendBundle"]))

check("src.metrics imports",
      lambda: __import__("src.metrics", fromlist=["compute_metrics"]))

check("src.io_utils imports",
      lambda: __import__("src.io_utils", fromlist=[
          "load_yaml", "save_json", "generate_manifest",
          "_make_run_id", "aggregate_runs"
      ]))

check("src.reservoir_adapter imports",
      lambda: __import__("src.reservoir_adapter", fromlist=["run_qrc_experiment"]))

check("src.runner imports",
      lambda: __import__("src.runner", fromlist=["execute_run"]))

check("src.plot_utils imports",
      lambda: __import__("src.plot_utils", fromlist=["make_all_plots"]))

check("configs/study_config.yaml readable", lambda: (
    __import__("src.io_utils", fromlist=["load_yaml"])
    .load_yaml(Path(__file__).parents[1] / "configs" / "study_config.yaml")
))

check("configs/paths.yaml readable", lambda: (
    __import__("src.io_utils", fromlist=["load_yaml"])
    .load_yaml(Path(__file__).parents[1] / "configs" / "paths.yaml")
))

print(f"\n{'ALL PASSED' if not errors else f'{len(errors)} FAILED'}\n")
if errors:
    sys.exit(1)
