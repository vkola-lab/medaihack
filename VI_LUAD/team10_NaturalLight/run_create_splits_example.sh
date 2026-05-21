#!/bin/bash

# run_create_splits_example.sh
# ----------------------------
# Creates 5-fold stratified patient-level cross-validation splits from the
# clinical label file. Run this once before training.
#
# Usage (from your project directory in an interactive session):
#   bash run_create_splits_example.sh
#
# This takes only a few seconds and requires no GPU.

module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/team10/vi_luad/bin/activate

# Splits are saved to splits/ inside your project directory.
# Make sure you run this script from /projectnb/medaihack/YOUR_TEAM/.
python create_splits.py \
    --label_file /projectnb/medaihack/VI_LUAD_Project/Clinical_Data/hackathon_label.txt \
    --splits_dir splits

# Optional: use a different number of folds or random seed
#   --n_folds    3
#   --random_seed 0
