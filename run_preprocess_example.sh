#!/bin/bash -l

#$ -N preprocess_vi_luad       # job name
#$ -P medaihack                # SCC project group
#$ -l h_rt=02:00:00            # wall-clock limit — adjust to your workload.
#                               #   Lower values get higher queue priority, so don't
#                               #   over-request. 2 hours covers the baseline cTransPath
#                               #   encoder on all 709 WSIs with a safe margin. If you
#                               #   swap in a larger encoder, increase this accordingly.
#                               #   If the job hits the limit mid-run, it is killed.
#$ -pe omp 16                  # 16 CPU cores (OpenMP parallel environment)
#$ -l gpus=1                   # 1 GPU
#$ -l gpu_c=8.0                # GPU compute capability >= 8.0 (L40S)
#$ -l gpu_m=20G                # minimum GPU memory: 20 GB
#$ -l mem_per_core=2G          # 2 GB RAM per core → 32 GB total
#$ -j y                        # merge stdout and stderr
#$ -o preprocess.log           # log file — after qsub, monitor with: tail -f preprocess.log

# Load conda and activate environment
module load miniconda
conda activate vi_luad

# Move to your project directory so relative paths resolve correctly
cd <your_project_dir>          # e.g. /projectnb/medaihack/<your_team_name>/

echo "=========================================="
echo "Job started: $(date)"
echo "Running on host: $(hostname)"
echo "GPU(s) assigned: $SGE_HGR_gpu"
echo "=========================================="

# Replace each <placeholder> with your actual path:
#   <your_feature_folder> e.g. /projectnb/medaihack/<your_team_name>/features
#   <your_splits_folder>  e.g. /projectnb/medaihack/<your_team_name>/splits
python starter_code/preprocess.py \
    --wsi_dir      /projectnb/medaihack/VI_LUAD_Project/WSI_Data/wsi \
    --label_file   /projectnb/medaihack/VI_LUAD_Project/Clinical_Data/hackathon_label.txt \
    --output_dir   <your_feature_folder> \
    --splits_dir   <your_splits_folder> \
    --ctranspath_weights /projectnb/medaihack/VI_LUAD_Project/ctranspath.pth \
    --patch_size   1024 \
    --step_size    1024 \
    --batch_size   512 \
    --num_workers  16 \
    --read_level   1

# Additional parameters not shown above (all have sensible defaults):
#   --n_folds       N    Number of cross-validation folds (default: 5)
#   --random_seed   N    Random seed for reproducible splits (default: 42)
#   --tissue_thresh F    Min tissue fraction to keep a patch, 0–1 (default: 0.5)
#   --splits_only        Skip feature extraction; only (re)generate split files
#
# Run `python starter_code/preprocess.py --help` for the full parameter list.

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
