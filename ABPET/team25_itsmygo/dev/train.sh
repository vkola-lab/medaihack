#!/bin/bash
# dev/train.sh — launcher for dev/train.py
# ─────────────────────────────────────────
# Usage:
#     bash dev/train.sh                               # uses dev/config/train.yaml
#     bash dev/train.sh dev/config/train_freeze.yaml  # custom config
#
# Every knob (paths, hyperparameters, loss, model variant, freeze_tracer_norm)
# lives in the YAML — edit that file, don't add CLI flags.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${1:-$SCRIPT_DIR/config/train.yaml}"

python3 "$SCRIPT_DIR/train.py" --config "$CONFIG"
