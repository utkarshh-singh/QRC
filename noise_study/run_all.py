#!/usr/bin/env python3
"""
run_all.py
----------
Orchestrates the full noise study with parallel execution.
Uses 6 worker processes (out of 8 cores) — one full experiment per process.

Usage
-----
cd QRC/noise_study
python run_all.py [--force-rebuild-data] [--dry-run] [--workers N]
"""
from __future__ import annotations
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data import build_dataset
from src.io_utils import (
    generate_manifest,
    load_manifest,
    load_yaml,
    save_manifest,
    load_json,
)
from src.runner import execute_run

ROOT = Path(__file__).parent


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )


def _run_one_worker(args: dict) -> tuple[str, bool]:
    """
    Top-level function for each worker process.
    Must be a module-level function (not a lambda) for pickling.
    Returns (run_id, success).
    """
    
    import os
    os.environ["OMP_NUM_THREADS"]        = "2"
    os.environ["QISKIT_NUM_THREADS"]     = "2"
    os.environ["OPENBLAS_NUM_THREADS"]   = "2"
    os.environ["MKL_NUM_THREADS"]        = "2"

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from src.io_utils import load_yaml
    from src.runner import execute_run

    ROOT = Path(__file__).parent
    study_cfg = load_yaml(ROOT / "configs" / "study_config.yaml")
    paths_cfg = load_yaml(ROOT / "configs" / "paths.yaml")
    runs_root = ROOT / Path(paths_cfg["runs_dir"])

    row      = args["row"]
    dataset  = args["dataset"]

    success = execute_run(
        row       = row,
        study_cfg = study_cfg,
        paths_cfg = paths_cfg,
        runs_root = runs_root,
        dataset   = dataset,
    )
    return row["run_id"], success


def main():
    parser = argparse.ArgumentParser(description="Run full QRC noise study.")
    parser.add_argument("--force-rebuild-data", action="store_true")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print manifest only; do not execute.")
    parser.add_argument("--workers",  type=int, default=6,
                        help="Number of parallel worker processes (default: 6).")
    args = parser.parse_args()

    setup_logging(ROOT / "logs" / "study.log")
    logger = logging.getLogger("run_all")

    study_cfg = load_yaml(ROOT / "configs" / "study_config.yaml")
    paths_cfg = load_yaml(ROOT / "configs" / "paths.yaml")
    runs_root = ROOT / paths_cfg["runs_dir"]

    # ── Dataset (built once, passed to all workers) ────────────
    logger.info("Preparing dataset…")
    dataset = build_dataset(
        study_cfg["dataset"],
        paths_cfg["data"],
        force_rebuild=args.force_rebuild_data,
    )
    logger.info(
        "Dataset ready: %d train / %d test windows.",
        len(dataset["X_train"]), len(dataset["X_test"]),
    )

    # ── Manifest ───────────────────────────────────────────────
    manifest_path = ROOT / paths_cfg["manifests"]["planned_runs"]
    if manifest_path.exists():
        logger.info("Loading existing manifest from %s", manifest_path)
        manifest = load_manifest(manifest_path)
    else:
        logger.info("Generating new manifest…")
        manifest = generate_manifest(study_cfg)
        save_manifest(manifest, manifest_path)

    logger.info("Total planned runs: %d", len(manifest))

    if args.dry_run:
        print(manifest.to_string())
        return

    # ── Filter out already-successful runs ─────────────────────
    pending_rows = []
    skipped = 0
    for _, row in manifest.iterrows():
        status_file = runs_root / row["run_id"] / "status.json"
        if status_file.exists():
            status = load_json(status_file)
            if status.get("status") == "success":
                skipped += 1
                continue
        pending_rows.append(row)

    logger.info(
        "Runs to execute: %d  |  Already succeeded (skipped): %d  |  Workers: %d",
        len(pending_rows), skipped, args.workers,
    )

    if not pending_rows:
        logger.info("All runs already succeeded. Nothing to do.")
        return

    # ── Parallel execution ─────────────────────────────────────
    # Each worker gets its own copy of the dataset (passed via args dict).
    # study_cfg and paths_cfg are reloaded inside each worker from disk
    # to avoid pickling issues with complex yaml objects.
    worker_args = [{"row": row, "dataset": dataset} for row in pending_rows]

    results = {"success": 0, "failed": 0, "skipped": skipped}
    failed_ids = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one_worker, wa): wa["row"]["run_id"]
            for wa in worker_args
        }
        with tqdm(total=len(futures), desc="Experiments", unit="run") as pbar:
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    _, success = future.result()
                    if success:
                        results["success"] += 1
                        pbar.set_postfix_str(f"✓ {run_id}")
                    else:
                        results["failed"] += 1
                        failed_ids.append(run_id)
                        pbar.set_postfix_str(f"✗ {run_id}")
                except Exception as exc:
                    results["failed"] += 1
                    failed_ids.append(run_id)
                    logger.error("Worker crash for %s: %s", run_id, exc)
                pbar.update(1)

    # ── Summary ────────────────────────────────────────────────
    logger.info(
        "\nStudy complete. ✓ Success=%d  ✗ Failed=%d  ⏭ Skipped=%d",
        results["success"], results["failed"], results["skipped"],
    )
    if failed_ids:
        logger.warning("Failed runs:\n  %s", "\n  ".join(failed_ids))
        logger.info("Re-run with: python run_all.py  (failed runs will retry automatically)")


if __name__ == "__main__":
    main()
