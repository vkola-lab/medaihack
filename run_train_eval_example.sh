#!/bin/bash -l

#$ -N train_vi_luad             # job name (shown in qstat)
#$ -P medaihack                 # SCC project group
#$ -l h_rt=01:00:00             # wall-clock limit — adjust to your workload.
#                               #   Lower values get higher queue priority, so don't
#                               #   over-request. The baseline MIL model finishes in
#                               #   ~5 minutes; 1 hour is enough for most approaches.
#                               #   Only increase if you are training a very large model
#                               #   or running many more epochs than the default.
#$ -pe omp 4                    # 4 CPU cores
#$ -l gpus=1                    # 1 GPU
#$ -l gpu_c=8.0                 # GPU compute capability >= 8.0 (L40S)
#$ -l gpu_m=8G                  # minimum GPU memory: 8 GB
#$ -l mem_per_core=4G           # 4 GB RAM per core → 16 GB total
#$ -j y                         # merge stdout and stderr
#$ -o train_eval.log            # log file — after qsub, monitor with: tail -f train_eval.log

# Load conda and activate environment
module load miniconda
conda activate vi_luad

# Move to your project directory so relative paths resolve correctly
cd <your_project_dir>           # e.g. /projectnb/medaihack/<your_team_name>/

echo "=========================================="
echo "Job started:  $(date)"
echo "Running on host: $(hostname)"
echo "GPU(s) assigned: $SGE_HGR_gpu"
echo "=========================================="

# Replace each <placeholder> with your actual path:
#   <your_feature_folder>    e.g. starter_code/features   (output of preprocess.py)
#   <your_splits_folder>     e.g. starter_code/splits     (output of preprocess.py)
#   <your_checkpoint_folder> e.g. starter_code/checkpoints
#   <your_predictions_folder> e.g. starter_code/predictions
python starter_code/train_eval.py \
    --features_dir <your_feature_folder> \
    --splits_dir   <your_splits_folder> \
    --save_dir     <your_checkpoint_folder> \
    --preds_dir    <your_predictions_folder> \
    --epochs       20 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --hidden_dim   256 \
    --dropout      0.25 \
    --eval_every   5 \
    --seed         42

# Additional parameters not shown above (all have sensible defaults):
#   --batch_size  N   Slides per gradient step (default: 1; keep at 1 for MIL)
#   --folds       N…  Run only specific folds, e.g. --folds 0 1 (default: all 5)
#
# Run `python starter_code/train_eval.py --help` for the full parameter list.

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
