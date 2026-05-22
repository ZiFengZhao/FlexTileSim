#!/bin/bash

echo "=== NoC Experiment ==="

./build.sh

python3 run_sweep.py --sweep

python3 run_sensitivity.py --sweep --plot

echo "Done. Results in results/"