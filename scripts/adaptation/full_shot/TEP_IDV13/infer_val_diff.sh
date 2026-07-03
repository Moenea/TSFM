#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

inference_args "$DIFF_VAL_MODEL_ID" "$DIFF_VAL_SPLIT" "$DIFF_SETTING" "$DIFF_VAL_SETTING" \
  --restore_diff_to_raw \
  --raw_split_file "$RAW_VAL_SPLIT" \
  --restore_target "$TARGET"
