#!/usr/bin/env bash
set -euo pipefail
# Timer-XL raw + diff experts (MS 5-var -> XMEAS10) for one SUBSET_RATIO.
# Produces 4 result dirs: raw/test, diff/test, raw/val, diff/val.
# Usage: SUBSET_RATIO=0.1 bash scripts/adaptation/xmeas10/run_timerxl.sh
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/_common.sh"

# idempotent: (re)write raw-aligned first differences for the 5-var CSVs
"$PYTHON_BIN" "$HERE/prepare_5var_diff.py"

echo "###### Timer-XL raw (MS 5-var) ratio=$SUBSET_RATIO ######"
timer_args_ms "$RAW_MODEL_ID" "$RAW_SPLIT"

echo "###### Timer-XL diff (MS 5-var) ratio=$SUBSET_RATIO ######"
timer_args_ms "$DIFF_MODEL_ID" "$DIFF_SPLIT" \
  --restore_diff_to_raw --raw_split_file "$RAW_SPLIT" --restore_target "$TARGET"

echo "###### infer val: raw ######"
inference_args_ms "$RAW_VAL_MODEL_ID" "$RAW_VAL_SPLIT" "$RAW_SETTING" "$RAW_VAL_SETTING"

echo "###### infer val: diff ######"
inference_args_ms "$DIFF_VAL_MODEL_ID" "$DIFF_VAL_SPLIT" "$DIFF_SETTING" "$DIFF_VAL_SETTING" \
  --restore_diff_to_raw --raw_split_file "$RAW_VAL_SPLIT" --restore_target "$TARGET"

echo "DONE Timer-XL ratio=$SUBSET_RATIO"
echo "  raw  test dir : results/$RAW_SETTING"
echo "  diff test dir : results/$DIFF_SETTING"
echo "  raw  val  dir : results/$RAW_VAL_SETTING"
echo "  diff val  dir : results/$DIFF_VAL_SETTING"
