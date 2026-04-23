#!/bin/bash
# eda/run_all.sh — Run the EDA suite (pre-train and/or post-train)
# ==============================================================
# Usage:
#     bash eda/run_all.sh                                # pre-train only (01–03)
#     bash eda/run_all.sh --pred_csv predictions.csv     # pre-train + post-train (01–04)
#     bash eda/run_all.sh --post_only --pred_csv predictions.csv  # post-train only (04)
#     bash eda/run_all.sh --cohort_csv data/cohort.csv   # include cohort plots in 03
#     bash eda/run_all.sh --tag baseline_no_film         # stamp all figures with label+UTC time
#
# Outputs:
#     results/eda/pre_train/01_centiloid_distribution/
#     results/eda/pre_train/02_tracer_comparison/
#     results/eda/pre_train/03_calibration_analysis/
#     results/eda/post_train/04_model_error_analysis/    (only with --pred_csv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────
TRAIN_CSV="${TRAIN_CSV:-$REPO_ROOT/data/train.csv}"
VAL_CSV="${VAL_CSV:-$REPO_ROOT/data/val.csv}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/eda}"
N_SAMPLES="${N_SAMPLES:-50}"
N_SLICES="${N_SLICES:-3}"
PRED_CSV=""
COHORT_CSV=""
POST_ONLY=false
TAG=""

# ── Parse flags ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_csv)   TRAIN_CSV="$2";   shift 2 ;;
        --val_csv)     VAL_CSV="$2";     shift 2 ;;
        --out_root)    OUT_ROOT="$2";    shift 2 ;;
        --pred_csv)    PRED_CSV="$2";    shift 2 ;;
        --cohort_csv)  COHORT_CSV="$2";  shift 2 ;;
        --n_samples)   N_SAMPLES="$2";   shift 2 ;;
        --n_slices)    N_SLICES="$2";    shift 2 ;;
        --tag)         TAG="$2";         shift 2 ;;
        --post_only)   POST_ONLY=true;   shift 1 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Common tag arg for every script invocation below
TAG_ARG=""
if [[ -n "$TAG" ]]; then
    TAG_ARG="--tag $TAG"
fi

PRE_DIR="$OUT_ROOT/pre_train"
POST_DIR="$OUT_ROOT/post_train"

echo "============================================================"
echo "  MYGO — EDA Suite"
echo "============================================================"
echo "  train_csv  : $TRAIN_CSV"
echo "  val_csv    : $VAL_CSV"
echo "  pre_train  : $PRE_DIR"
echo "  post_train : $POST_DIR"
echo "  pred_csv   : ${PRED_CSV:-<none — post-train skipped>}"
echo "  cohort_csv : ${COHORT_CSV:-<none — cohort plots skipped>}"
echo "  tag        : ${TAG:-<auto — derived per script>}"
echo "  post_only  : $POST_ONLY"
echo "============================================================"
echo ""

FAILED=0
RAN=0

# ══════════════════════════════════════════════════════════════
#  PRE-TRAIN  (scripts 01–03: data analysis, no model needed)
# ══════════════════════════════════════════════════════════════

if [[ "$POST_ONLY" == false ]]; then

    # ── 01: Centiloid distribution ────────────────────────────
    echo ">>> [pre-train] 01_centiloid_distribution.py"
    python3 "$SCRIPT_DIR/01_centiloid_distribution.py" \
        --train_csv "$TRAIN_CSV" \
        --val_csv   "$VAL_CSV" \
        --out_dir   "$PRE_DIR/01_centiloid_distribution" \
        $TAG_ARG \
        || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
    RAN=$((RAN+1))
    echo ""

    # ── 02: Tracer comparison ─────────────────────────────────
    echo ">>> [pre-train] 02_tracer_comparison.py"
    python3 "$SCRIPT_DIR/02_tracer_comparison.py" \
        --train_csv "$TRAIN_CSV" \
        --val_csv   "$VAL_CSV" \
        --out_dir   "$PRE_DIR/02_tracer_comparison" \
        --n_slices  "$N_SLICES" \
        $TAG_ARG \
        || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
    RAN=$((RAN+1))
    echo ""

    # ── 03: Calibration analysis ──────────────────────────────
    echo ">>> [pre-train] 03_calibration_analysis.py"
    COHORT_ARG=""
    if [[ -n "$COHORT_CSV" ]]; then
        COHORT_ARG="--cohort_csv $COHORT_CSV"
    fi
    python3 "$SCRIPT_DIR/03_calibration_analysis.py" \
        --train_csv  "$TRAIN_CSV" \
        --val_csv    "$VAL_CSV" \
        --out_dir    "$PRE_DIR/03_calibration_analysis" \
        --n_samples  "$N_SAMPLES" \
        $COHORT_ARG \
        $TAG_ARG \
        || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
    RAN=$((RAN+1))
    echo ""

fi

# ══════════════════════════════════════════════════════════════
#  POST-TRAIN  (script 04: model error analysis, needs preds)
# ══════════════════════════════════════════════════════════════

if [[ -n "$PRED_CSV" ]]; then

    echo ">>> [post-train] 04_model_error_analysis.py"
    python3 "$SCRIPT_DIR/04_model_error_analysis.py" \
        --val_csv   "$VAL_CSV" \
        --pred_csv  "$PRED_CSV" \
        --out_dir   "$POST_DIR/04_model_error_analysis" \
        $TAG_ARG \
        || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
    RAN=$((RAN+1))
    echo ""

else
    if [[ "$POST_ONLY" == true ]]; then
        echo "  ✗ --post_only requires --pred_csv"
        exit 1
    fi
    echo "  · Skipping post-train EDA (no --pred_csv provided)"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────
echo "============================================================"
if [[ $FAILED -eq 0 ]]; then
    echo "  ✓ $RAN script(s) passed"
else
    echo "  ✗ $FAILED of $RAN script(s) failed"
fi
echo "  pre-train  → $PRE_DIR"
echo "  post-train → $POST_DIR"
echo "============================================================"

exit $FAILED
