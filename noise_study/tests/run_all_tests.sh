#!/usr/bin/env bash
# Run all test layers in order.
# Usage:  cd QRC/noise_study && bash tests/run_all_tests.sh

set -e
cd "$(dirname "$0")/.."

echo ""
echo "========================================="
echo "  QRC Noise Study — Test Suite"
echo "========================================="

run_test() {
    echo ""
    python "tests/$1"
    echo ""
}

run_test test_imports.py
run_test test_units.py
run_test test_noise_models.py
run_test test_integration_dry.py
run_test test_adapter.py

echo ""
echo "========================================="
echo "  All test layers passed."
echo "========================================="
