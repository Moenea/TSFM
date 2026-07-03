#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

inference_args "$DIFF_MODEL_ID" "$DIFF_SPLIT" "$DIFF_SETTING" "$DIFF_SETTING" \
  --restore_diff_to_raw \
  --raw_split_file "$RAW_SPLIT" \
  --restore_target "$TARGET"
