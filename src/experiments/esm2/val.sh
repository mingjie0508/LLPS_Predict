#!/bin/bash

# absolute path to this script
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
EXPERIMENT_DIR="src/experiments/esm2"

# get the project main directory
SCRIPT_DIR="$SCRIPT_PATH"

for i in $(seq 1 4);
do
    SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"
done

# change directory to project main directory
cd $SCRIPT_DIR

# create output and logging directories if not exist
OUTPUT_DIR="$EXPERIMENT_DIR/output"
LOGGING_DIR="$EXPERIMENT_DIR/logging"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d "$LOGGING_DIR" ]; then
    mkdir -p "$LOGGING_DIR"
fi

# run script
python -m src.scripts.test \
    --config $EXPERIMENT_DIR/val_config.yml \
    --data_path data/validation/llps_vs_non_llps_val.csv \
    --score_path $EXPERIMENT_DIR/output/esm2_650m_baseline_h4_l1_val_score.csv \
    > $EXPERIMENT_DIR/logging/esm2_650m_baseline_h4_l1_val.log

python -m src.scripts.test \
    --config $EXPERIMENT_DIR/val_config.yml \
    --data_path data/validation/driver_vs_non_llps_val.csv \
    --score_path $EXPERIMENT_DIR/output/esm2_650m_baseline_h4_l1_val_driver_score.csv \
    > $EXPERIMENT_DIR/logging/esm2_650m_baseline_h4_l1_val_driver.log

python -m src.scripts.test \
    --config $EXPERIMENT_DIR/val_config.yml \
    --data_path data/validation/partner_vs_non_llps_val.csv \
    --score_path $EXPERIMENT_DIR/output/esm2_650m_baseline_h4_l1_val_partner_score.csv \
    > $EXPERIMENT_DIR/logging/esm2_650m_baseline_h4_l1_val_partner.log
