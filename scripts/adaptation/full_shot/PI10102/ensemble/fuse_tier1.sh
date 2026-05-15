#!/usr/bin/env bash
# Tier 1 gate ensemble: per-step closed-form alpha(t) fused between
# source A (DIFF-to-ABS Timer-XL by default) and source B (raw Timer-XL).
# Fit on 3 transitions, predict on the held-out 4th (leave-one-out across 4 folds).
#
# Output: results/ensemble_${OUTPUT_NAME}_test_0/{pred.npy, true.npy, fit_log.json}

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/aicode/miniconda3/envs/tsfm/bin/python}
SOURCE_A=${SOURCE_A:-Timer-XL-MS-DIFF}
SOURCE_B=${SOURCE_B:-Timer-XL-MS}
OUTPUT_NAME=${OUTPUT_NAME:-Gate-T1-TXL-MS}

cd /home/aicode/sherwin/TSFM

"$PYTHON_BIN" -u scripts/adaptation/full_shot/PI10102/ensemble/fuse_predictions.py \
  --source_a_name "$SOURCE_A" \
  --source_b_name "$SOURCE_B" \
  --method tier1 \
  --output_name "$OUTPUT_NAME"
