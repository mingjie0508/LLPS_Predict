#!/bin/bash

# absolute path to this script
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
EXPERIMENT_DIR="src/experiments/conditions"

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
python -m src.scripts.test_conditions \
    --config $EXPERIMENT_DIR/val_config.yml \
    --data_path data/experimental_condition/LLPSDB2_conditions_LAF1.csv \
    --score_path $EXPERIMENT_DIR/output/LLPSDB2_conditions_LAF1_score.csv \
    > $EXPERIMENT_DIR/logging/LLPSDB2_conditions_LAF1.log
