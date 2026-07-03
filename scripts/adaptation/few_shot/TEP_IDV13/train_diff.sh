#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Idempotent: (re)writes raw-aligned first differences under csv_diff/.
"$PYTHON_BIN" "$(dirname "$0")/prepare_diff.py"

timer_args "$DIFF_MODEL_ID" "$DIFF_SPLIT" \
  --restore_diff_to_raw \
  --raw_split_file "$RAW_SPLIT" \
  --restore_target "$TARGET"
