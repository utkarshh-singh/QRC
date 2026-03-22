#!/usr/bin/env python3
"""
run_noise_sweep.py
------------------
Parametric noise sweep study. Runs sequentially (1 worker) —
each run is ~20 min, 30 runs total = ~10 hours.

Usage
-----
cd QRC/noise_study
python run_noise_sweep.py [--dry-run] [--workers N]
"""
from __future__ import annotations

import argparse, json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from src.data        import build_dataset
from src.io_utils    import load_yaml
from src.metrics     import compute_metrics
from src.parametric_noise_models import get_all_bundles, ParametricBundle

ROOT = Path(__file__).parent


def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "noise_sweep.log"),
        ],
    )


def run_one(bundle: ParametricBundle, dataset: dict,
            study_cfg: dict, run_dir: Path) -> dict:
    """Run one parametric experiment and save artifacts."""
    import numpy as np
    from src.reservoir_adapter import run_qrc_experiment

    run_dir.mkdir(parents=True, exist_ok=True)

    # Inject bundle into study_cfg format expected by run_qrc_experiment
    t0 = time.perf_counter()
    result = run_qrc_experiment(
        X_train   = dataset["X_train"],
        y_train   = dataset["y_train"],
        X_test    = dataset["X_test"],
        y_test    = dataset["y_test"],
        study_cfg = study_cfg,
        bundle    = bundle,
    )
    elapsed = time.perf_counter() - t0

    y_pred = result["y_test_pred"]
    y_true = result["y_test_true"]
    metrics = compute_metrics(y_true, y_pred)

    # Save artifacts
    metadata = {
        "run_id":      bundle.label,
        "noise_type":  bundle.noise_type,
        "noise_param": bundle.noise_param,
        "label":       bundle.label,
        "shots":       bundle.shots,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    import pandas as pd
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"y_true": y_true.ravel(),
                  "y_pred": y_pred.ravel()}).to_csv(
        run_dir / "predictions.csv", index=False)
    (run_dir / "status.json").write_text(json.dumps({"status": "success"}))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("run_noise_sweep")

    cfg = load_yaml(ROOT / "configs" / "noise_sweep_config.yaml")
    paths_cfg = load_yaml(ROOT / "configs" / "paths.yaml")

    dataset = build_dataset(cfg["dataset"], paths_cfg["data"])
    logger.info("Dataset: %d train / %d test",
                len(dataset["X_train"]), len(dataset["X_test"]))

    bundles = get_all_bundles(cfg)
    logger.info("Total parametric runs: %d", len(bundles))

    if args.dry_run:
        for b in bundles:
            print(f"  {b.label:<35}  noise_type={b.noise_type}  param={b.noise_param}")
        return

    runs_root = ROOT / "runs_noise_sweep"
    results = []

    for i, bundle in enumerate(bundles):
        run_dir = runs_root / bundle.label
        status_file = run_dir / "status.json"

        if status_file.exists():
            status = json.loads(status_file.read_text())
            if status.get("status") == "success":
                logger.info("SKIP %s (already done)", bundle.label)
                continue

        logger.info("[%d/%d] START %s", i+1, len(bundles), bundle.label)
        try:
            metrics = run_one(bundle, dataset, cfg, run_dir)
            logger.info("  DONE  RMSE=%.4f  R²=%.4f", metrics["rmse"], metrics["r2"])
            results.append({"label": bundle.label, **metrics})
        except Exception as e:
            logger.error("  FAILED %s: %s", bundle.label, e)

    # Save summary CSV
    if results:
        import pandas as pd
        summary = pd.DataFrame(results)
        out = ROOT / "results" / "aggregated" / "noise_sweep_results.csv"
        summary.to_csv(out, index=False)
        logger.info("Summary saved → %s", out)


if __name__ == "__main__":
    main()
