#!/usr/bin/env python3
"""
aggregate_results.py
--------------------
Collect all completed run results into:
  results/aggregated/master_results.csv   — one row per run
  results/aggregated/grouped_results.csv  — mean ± std grouped by
                                            (experiment_type, noise_type, shots)

Usage
-----
    cd QRC/noise_study
    python aggregate_results.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.io_utils import aggregate_runs, load_yaml

ROOT = Path(__file__).parent
METRIC_COLS = ["mse", "rmse", "mae", "r2", "pearson"]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
    )
    logger = logging.getLogger("aggregate_results")

    paths_cfg = load_yaml(ROOT / "configs" / "paths.yaml")
    runs_root = ROOT / paths_cfg["runs_dir"]
    agg_dir = ROOT / paths_cfg["results"]["aggregated_dir"]
    agg_dir.mkdir(parents=True, exist_ok=True)

    master = aggregate_runs(runs_root)

    if master.empty:
        logger.error("No successful runs to aggregate. Run the study first.")
        sys.exit(1)

    master_path = ROOT / paths_cfg["results"]["master_csv"]
    master.to_csv(master_path, index=False)
    logger.info("Master CSV saved (%d rows): %s", len(master), master_path)

    # Grouped summary: mean and std over seeds
    group_keys = ["experiment_type", "noise_type", "shots"]
    available_keys = [k for k in group_keys if k in master.columns]
    available_metrics = [c for c in METRIC_COLS if c in master.columns]

    if available_keys and available_metrics:
        grouped = (
            master.groupby(available_keys)[available_metrics]
            .agg(["mean", "std"])
        )
        # Flatten multi-level columns: (rmse, mean) -> rmse_mean
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        grouped = grouped.reset_index()

        grouped_path = ROOT / paths_cfg["results"]["grouped_csv"]
        grouped.to_csv(grouped_path, index=False)
        logger.info("Grouped CSV saved: %s", grouped_path)

        # Print a quick summary table to stdout
        print("\n=== Grouped Results (mean over seeds) ===")
        cols_to_show = available_keys + [f"{m}_mean" for m in available_metrics]
        cols_to_show = [c for c in cols_to_show if c in grouped.columns]
        print(grouped[cols_to_show].to_string(index=False))
    else:
        logger.warning("Could not compute grouped stats (missing columns).")


if __name__ == "__main__":
    main()
