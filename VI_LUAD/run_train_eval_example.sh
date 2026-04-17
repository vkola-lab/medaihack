#!/bin/bash -l

#$ -N vi_luad_train_team15
#$ -P medaihack
#$ -j y
#$ -o train_eval_team15.log
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_type=L40S

# =============================================================================

# ---- Fill in your team name ----
TEAM=team15

# --- cd to the team's project directory --------------------------------------
cd /projectnb/medaihack/$TEAM || { echo "[ERROR] Cannot cd to /projectnb/medaihack/$TEAM"; exit 1; }
echo "[INFO] Working directory: $(pwd)"

# --- Load modules and activate the team's virtual environment ----------------
module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/$TEAM/vi_luad/bin/activate
echo "[INFO] Virtual environment activated."
nvidia-smi || echo "[WARN] nvidia-smi unavailable"

FEATURES_DIR=/projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed
SPLITS_DIR=starter_code/splits
CHECKPOINTS_DIR=checkpoints
PREDICTIONS_DIR=predictions

echo "[INFO] Creating splits..."
python starter_code/create_splits.py \
    --splits_dir    $SPLITS_DIR \
    --n_folds       5 \
    --val_frac      0.15 \
    --include_nontumor

echo "[INFO] Starting training..."
python -u starter_code/train_eval.py \
    --features_dir     $FEATURES_DIR \
    --splits_dir       $SPLITS_DIR \
    --save_dir         $CHECKPOINTS_DIR \
    --preds_dir        $PREDICTIONS_DIR \
    --epochs           50 \
    --patience         10 \
    --lr               1e-4 \
    --weight_decay     1e-4 \
    --label_smoothing  0.05 \
    --pos_weight       auto \
    --branch_ce_weight 0.5 \
    --entropy_weight   0.01 \
    --hidden_dim       512 \
    --dropout          0.25 \
    --n_branches       5 \
    --top_k            10 \
    --mask_prob        0.6 \
    --use_pe           True \
    --pe_dim           64 \
    --clip_eps         0.02 \
    --n_seeds          5 \
    --base_seed        42 \
    --num_workers      8

echo "[INFO] train_eval.py finished with exit code $?"
