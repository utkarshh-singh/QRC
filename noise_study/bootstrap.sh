#!/usr/bin/env bash
# FILE: QRC/noise_study/bootstrap.sh
# Run from QRC/ :  bash noise_study/bootstrap.sh

ROOT="noise_study"

mkdir -p $ROOT/{configs,src,manifests,data/{raw,processed},runs,results/{aggregated,figures,tables},logs}

touch $ROOT/README.md
touch $ROOT/requirements.txt
touch $ROOT/.gitignore
touch $ROOT/run_all.py
touch $ROOT/run_one.py
touch $ROOT/aggregate_results.py
touch $ROOT/make_plots.py
touch $ROOT/configs/study_config.yaml
touch $ROOT/configs/paths.yaml
touch $ROOT/src/__init__.py
touch $ROOT/src/data.py
touch $ROOT/src/noise_models.py
touch $ROOT/src/reservoir_adapter.py
touch $ROOT/src/runner.py
touch $ROOT/src/metrics.py
touch $ROOT/src/io_utils.py
touch $ROOT/src/plot_utils.py

echo "Scaffold created under $ROOT/"