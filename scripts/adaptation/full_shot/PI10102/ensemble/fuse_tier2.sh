#!/usr/bin/env bash
# Tier 2 gate ensemble: context-only MLP gate -> sigma(t) in [0,1]^15.
# Fused = sigma * A + (1 - sigma) * B. Leave-one-transition-out across 4 folds.
#
# Output: results/ensemble_${OUTPUT_NAME}_test_0/{pred.npy, true.npy, fit_log.json}

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/aicode/miniconda3/envs/tsfm/bin/python}
SOURCE_A=${SOURCE_A:-Timer-XL-MS-DIFF}
SOURCE_B=${SOURCE_B:-Timer-XL-MS}
OUTPUT_NAME=${OUTPUT_NAME:-Gate-T2-TXL-MS}
EPOCHS=${EPOCHS:-300}
HIDDEN=${HIDDEN:-32}
LR=${LR:-1e-3}
WD=${WD:-1e-4}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cpu}

cd /home/aicode/sherwin/TSFM

"$PYTHON_BIN" -u scripts/adaptation/full_shot/PI10102/ensemble/fuse_predictions.py \
  --source_a_name "$SOURCE_A" \
  --source_b_name "$SOURCE_B" \
  --method tier2 \
  --output_name "$OUTPUT_NAME" \
  --epochs "$EPOCHS" \
  --hidden "$HIDDEN" \
  --lr "$LR" \
  --weight_decay "$WD" \
  --seed "$SEED" \
  --device "$DEVICE"
