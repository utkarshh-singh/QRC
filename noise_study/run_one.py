#!/usr/bin/env python3
"""
run_one.py
----------
Execute a single run by specifying its parameters directly.
Useful for testing, debugging, or re-running a failed experiment.

Usage
-----
    cd QRC/noise_study
    python run_one.py --noise-type ideal --shots none --seed 0
    python run_one.py --noise-type shot_only --shots 1024 --seed 1
    python run_one.py --noise-type full_backend --shots 1024 --seed 2
    python run_one.py --run-id ablation_full_backend_s1024_seed0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data import build_dataset
from src.io_utils import (
    generate_manifest,
    load_json,
    load_manifest,
    load_yaml,
)
from src.runner import execute_run

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Execute a single QRC noise study run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str,
                       help="Exact run_id from planned_runs.csv.")
    group.add_argument("--noise-type", type=str,
                       choices=["ideal", "shot_only", "readout_only",
                                "single_qubit_only", "two_qubit_only",
                                "relaxation_only", "full_backend"],
                       help="Noise type to run.")

    parser.add_argument("--shots", type=str, default="1024",
                        help="Number of shots, or 'none' for statevector.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    logger = logging.getLogger("run_one")

    study_cfg = load_yaml(ROOT / "configs" / "study_config.yaml")
    paths_cfg = load_yaml(ROOT / "configs" / "paths.yaml")
    runs_root = ROOT / paths_cfg["runs_dir"]

    dataset = build_dataset(study_cfg["dataset"], paths_cfg["data"])

    if args.run_id:
        manifest_path = ROOT / paths_cfg["manifests"]["planned_runs"]
        if not manifest_path.exists():
            manifest = generate_manifest(study_cfg)
        else:
            manifest = load_manifest(manifest_path)

        matches = manifest[manifest["run_id"] == args.run_id]
        if matches.empty:
            logger.error("run_id '%s' not found in manifest.", args.run_id)
            sys.exit(1)
        row = matches.iloc[0]
    else:
        shots = None if args.shots.lower() == "none" else int(args.shots)
        exp_type = "ideal" if args.noise_type == "ideal" else "manual"
        from src.io_utils import _make_run_id
        run_id = _make_run_id(exp_type, shots, args.seed, args.noise_type)
        row = pd.Series({
            "run_id": run_id,
            "experiment_type": exp_type,
            "noise_type": args.noise_type,
            "shots": shots,
            "seed": args.seed,
        })

    logger.info("Running: %s", row["run_id"])
    success = execute_run(
        row=row,
        study_cfg=study_cfg,
        paths_cfg=paths_cfg,
        runs_root=runs_root,
        dataset=dataset,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
