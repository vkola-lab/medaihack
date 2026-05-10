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
# Participants: hardcode the full path to YOUR TEAM's venv below.
# This is the venv the organizers will activate at evaluation time.
VENV_ACTIVATE="/projectnb/medaihack/team25/venv_name/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: venv activate script not found at:" >&2
    echo "       $VENV_ACTIVATE" >&2
    echo "       Edit predict.sh and point VENV_ACTIVATE at your team venv." >&2
    exit 1
fi
source "$VENV_ACTIVATE"

INPUT="$1"
CHECKPOINT="${2:-$SCRIPT_DIR/checkpoints/best_model.pt}"
OUTPUT="${3:-predictions.csv}"

python3 "$SCRIPT_DIR/dev/predict.py" \
    --csv        "$INPUT" \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUTPUT"