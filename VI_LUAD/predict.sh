#!/bin/bash -l

###############################################################################
#                        LEADERBOARD PREDICTION JOB
###############################################################################
#
# This is the script that determines your team's leaderboard score.
# The organizers will run this script on your behalf against a held-out
# external test set. You do NOT need to run this script yourself (you
# will not have access to the test data).
#
# Your job: fill in the settings below and make sure predict.py loads
# your model correctly. The organizers will handle the rest.
#
# Your final score will be computed by the organizers on the full external
# test set after the hackathon ends.
#
# IMPORTANT:
#   1. Replace YOUR_TEAM below with your team's directory name (same as in README).
#   2. Set CHECKPOINT to the path of your model checkpoint file.
#   3. If you changed the model architecture, update load_checkpoint() in
#      predict.py to import and instantiate your model correctly.
#
# You can use any checkpoint you like:
#   - The best fold checkpoint from train_eval.py (e.g. checkpoints/fold_0.pth)
#   - A model trained on all training data without cross-validation
#
###############################################################################

# ======================= SGE DIRECTIVES ======================================

#$ -N leaderboard_predict
#$ -P medaihack
#$ -j y
#$ -o predict.log
#$ -l h_rt=01:00:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=8.9

# =============================================================================

# ---- Fill in your settings ----
TEAM=YOUR_TEAM                    # same as YOUR_TEAM in README
CHECKPOINT=PATH/TO/YOUR_CKPT.pth

# --- cd to the team's project directory --------------------------------------

cd /projectnb/medaihack/$TEAM || { echo "[ERROR] Cannot cd to /projectnb/medaihack/$TEAM"; exit 1; }
echo "[INFO] Working directory: $(pwd)"

# --- Load modules and activate the team's virtual environment ----------------

module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/$TEAM/vi_luad/bin/activate

echo "[INFO] Virtual environment '/projectnb/medaihack/$TEAM/vi_luad' activated."

# --- Run prediction ----------------------------------------------------------

echo "[INFO] Running predict.py ..."
python starter_code/predict.py \
    --team        $TEAM \
    --checkpoint  $CHECKPOINT

echo "[INFO] predict.py finished with exit code $?"
