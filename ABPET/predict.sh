#!/bin/bash
# predict.sh — Predict centiloid scores for samples in a CSV
# ===========================================================
# Usage:
#     bash predict.sh <input.csv> [checkpoint.pt] [output.csv]
#
# Arguments:
#     input.csv       — CSV with npy_path and TRACER.AMY columns
#     checkpoint.pt   — (optional) path to trained model checkpoint (default: checkpoints/best_model.pt)
#     output.csv      — (optional) path for predictions (default: predictions.csv)
#
# Example:
#     bash predict.sh /projectnb/medaihack/ABPET/data/val.csv
#     bash predict.sh /projectnb/medaihack/ABPET/data/val.csv /projectnb/medaihack/ABPET/medaihack/ABPET/checkpoints/best_model.pt my_predictions.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: bash predict.sh <input.csv> [checkpoint.pt] [output.csv]"
    echo ""
    echo "Predict centiloid scores for each sample in the input CSV."
    exit 1
fi

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate" # Participants, please hardcode the path to YOUR TEAM's desired virtual env. This is the venv we will activate for evaluation.

INPUT="$1"
CHECKPOINT="${2:-$SCRIPT_DIR/checkpoints/best_model.pt}"
OUTPUT="${3:-predictions.csv}"

python3 "$SCRIPT_DIR/predict.py" \
    --csv        "$INPUT" \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUTPUT"
