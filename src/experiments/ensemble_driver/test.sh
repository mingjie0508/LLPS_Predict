#!/bin/bash

# absolute path to this script
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
EXPERIMENT_DIR="src/experiments/ensemble_driver"

# script's parent directory
SCRIPT_DIR="$SCRIPT_PATH"

# get the project main directory
for i in $(seq 1 4);
do
    SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"
done

# change directory to project main directory
cd $SCRIPT_DIR

# run script
python -m src.scripts.test_ensemble \
    --config $EXPERIMENT_DIR/test_config.yml \
    > $EXPERIMENT_DIR/logging/ensemble_driver_baseline_h4_l1_test.log
