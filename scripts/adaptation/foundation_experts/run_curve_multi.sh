#!/usr/bin/env bash
# run_curve_multi.sh — sweep few-shot ratios for the 3-expert FM gate
# (Timer-XL diff, Timer-XL raw, Time-MoE zero-shot) and collect a curve.
#
# Usage:
#   bash scripts/adaptation/foundation_experts/run_curve_multi.sh
#
# Env vars (all optional, defaults shown):
#   RATIOS   — space-separated list of ratios (default: all 7)
#   TSFM_PY  — Python interpreter (default: tsfm conda env)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
TSFM_PY=${TSFM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
SUMMARY=results/TEP_IDV13_XMEAS07_FM_Summary
SUFFIX=_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RATIOS=${RATIOS:-"0.01 0.02 0.05 0.1 0.25 0.5 1.0"}

cd "$ROOT"
mkdir -p "$SUMMARY"

for r in $RATIOS; do
  TAG=$(printf 'r%s' "$r" | tr '.' 'p')
  echo ""
  echo "###### FM gate ratio=$r  tag=$TAG ######"

  # Timer-XL dirs for this ratio
  PREFIX=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}
  DIFF_TEST="${PREFIX}_DIFF${SUFFIX}"
  RAW_TEST="${PREFIX}${SUFFIX}"
  DIFF_VAL="${PREFIX}_DIFF_val${SUFFIX}"
  RAW_VAL="${PREFIX}_val${SUFFIX}"

  # Verify Timer-XL dirs exist
  for d in "$DIFF_TEST" "$RAW_TEST" "$DIFF_VAL" "$RAW_VAL"; do
    if [ ! -d "$d" ]; then
      echo "ERROR: missing expected dir: $d" >&2
      exit 1
    fi
  done

  # Time-MoE is zero-shot — constant across all ratios
  TIME_MOE_TEST=results/fm_time_moe_zeroshot_test
  TIME_MOE_VAL=results/fm_time_moe_zeroshot_val
  for d in "$TIME_MOE_TEST" "$TIME_MOE_VAL"; do
    if [ ! -d "$d" ]; then
      echo "ERROR: missing Time-MoE zero-shot dir: $d" >&2
      exit 1
    fi
  done

  GATE=results/ensemble_Gate-multi_${TAG}_test

  # Step 1: fuse gate
  echo "[curve] Fusing gate for ratio=$r ..."
  "$TSFM_PY" "$HERE/fuse_gate_multi.py" \
    --expert "diff:${DIFF_TEST}" \
    --expert "raw:${RAW_TEST}" \
    --expert "time_moe:${TIME_MOE_TEST}" \
    --val-expert "diff:${DIFF_VAL}" \
    --val-expert "raw:${RAW_VAL}" \
    --val-expert "time_moe:${TIME_MOE_VAL}" \
    --data-root "$DATA_ROOT" \
    --train-split setting/TEP_IDV13_XMEAS07.yaml \
    --val-split   setting/TEP_IDV13_XMEAS07_val.yaml \
    --test-split  setting/TEP_IDV13_XMEAS07.yaml \
    --target "$TARGET" \
    --output-dir "$GATE"

  # Step 2: evaluate all experts + gate
  echo "[curve] Evaluating ratio=$r ..."
  "$TSFM_PY" "$HERE/evaluate_multi.py" \
    --expert "diff:${DIFF_TEST}" \
    --expert "raw:${RAW_TEST}" \
    --expert "time_moe:${TIME_MOE_TEST}" \
    --expert "gate:${GATE}" \
    --data-root "$DATA_ROOT" \
    --split setting/TEP_IDV13_XMEAS07.yaml \
    --target "$TARGET" \
    --output "${SUMMARY}/metrics_${TAG}.json"

  echo "[curve] Done ratio=$r -> ${SUMMARY}/metrics_${TAG}.json"
done

echo ""
echo "###### Collecting curve ######"
"$TSFM_PY" "$HERE/collect_curve_multi.py" \
  --ratios $RATIOS \
  --summary-dir "$ROOT/$SUMMARY"

echo ""
echo "[run_curve_multi] All done. Outputs in $SUMMARY/"
