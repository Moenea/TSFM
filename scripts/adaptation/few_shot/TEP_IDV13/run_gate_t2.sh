#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

GATE_DIR=$ROOT/results/ensemble_Gate-T2-TEP-IDV13-XMEAS07-S_few_${RATIO_TAG}_test_0
SUMMARY_DIR=$ROOT/results/TEP_IDV13_XMEAS07_FewShot_Summary

"$PYTHON_BIN" "$(dirname "$0")/fuse_gate_t2.py" \
  --data-root "$DATA_ROOT" \
  --raw-train-split "$RAW_SPLIT" \
  --raw-val-split "$RAW_VAL_SPLIT" \
  --raw-test-split "$RAW_SPLIT" \
  --target "$TARGET" \
  --val-diff-dir "$ROOT/results/$DIFF_VAL_SETTING" \
  --val-raw-dir "$ROOT/results/$RAW_VAL_SETTING" \
  --test-diff-dir "$ROOT/results/$DIFF_SETTING" \
  --test-raw-dir "$ROOT/results/$RAW_SETTING" \
  --output-dir "$GATE_DIR" \
  --seq-len "$SEQ_LEN" \
  --pred-len "$PRED_LEN" \
  --horizon 15

"$PYTHON_BIN" "$(dirname "$0")/evaluate.py" \
  --data-root "$DATA_ROOT" \
  --split "$RAW_SPLIT" \
  --target "$TARGET" \
  --raw-dir "$ROOT/results/$RAW_SETTING" \
  --diff-dir "$ROOT/results/$DIFF_SETTING" \
  --gate-dir "$GATE_DIR" \
  --output "$SUMMARY_DIR/metrics_${RATIO_TAG}.json" \
  --seq-len "$SEQ_LEN" \
  --pred-len "$PRED_LEN" \
  --horizon 15
