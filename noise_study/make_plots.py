#!/usr/bin/env python3
"""
make_plots.py
-------------
Generate all publication-ready figures from aggregated results.

Plots saved to:  results/figures/
  performance_vs_shots_rmse.pdf
  performance_vs_shots_r2.pdf
  noise_type_comparison_rmse.pdf
  noise_type_comparison_r2.pdf
  prediction_trace_<run_id>.pdf

Usage
-----
    cd QRC/noise_study
    python make_plots.py
    python make_plots.py --trace-run-id ideal_seed0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.io_utils import load_yaml
from src.plot_utils import make_all_plots

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Generate noise study figures.")
    parser.add_argument("--trace-run-id", type=str, default=None,
                        help="run_id to use for prediction trace plot.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
    )
    logger = logging.getLogger("make_plots")

    paths_cfg = load_yaml(ROOT / "configs" / "paths.yaml")
    master_csv = ROOT / paths_cfg["results"]["master_csv"]

    if not master_csv.exists():
        logger.error(
            "master_results.csv not found at %s. "
            "Run aggregate_results.py first.", master_csv
        )
        sys.exit(1)

    master = pd.read_csv(master_csv)
    logger.info("Loaded %d rows from %s", len(master), master_csv)

    make_all_plots(
        master_df=master,
        runs_root=ROOT / paths_cfg["runs_dir"],
        figures_dir=ROOT / paths_cfg["results"]["figures_dir"],
        trace_run_id=args.trace_run_id,
    )
    logger.info("All plots saved to %s", paths_cfg["results"]["figures_dir"])


if __name__ == "__main__":
    main()
