#!/bin/bash

# run_train_eval_example.sh
# -------------------------
# Trains the baseline MIL model with 5-fold cross-validation and evaluates it.
# The baseline finishes in ~5 minutes on a GPU interactive session.
#
# Usage (from your project directory in an interactive GPU session):
#   bash run_train_eval_example.sh

module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/team10/vi_luad/bin/activate

# Pre-extracted UNI2-h features (shared, read-only — do not copy these):
FEATURES_DIR=/projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed

# Paths under your project directory (run this script from /projectnb/medaihack/team10/):
SPLITS_DIR=splits
CHECKPOINTS_DIR=checkpoints
PREDICTIONS_DIR=predictions

python train_eval.py \
    --features_dir $FEATURES_DIR \
    --splits_dir   $SPLITS_DIR \
    --save_dir     $CHECKPOINTS_DIR \
    --preds_dir    $PREDICTIONS_DIR \
    --epochs       20 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --hidden_dim   256 \
    --dropout      0.25 \
    --eval_every   5 \
    --seed         42

# Additional parameters (all have sensible defaults):
#   --batch_size  N   Slides per gradient step (default: 1; keep at 1 for MIL)
#   --folds       N…  Run only specific folds, e.g. --folds 0 1 (default: all 5)
#
# Run `python train_eval.py --help` for the full parameter list.
