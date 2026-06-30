#!/usr/bin/env bash
set -euo pipefail
# Time-MoE expert: fit on the few-shot subset, then infer val (Run8) and test (Run9-10).
HERE=$(cd "$(dirname "$0")" && pwd)
FM_PY=${FM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
RATIO=${SUBSET_RATIO:-1.0}
TAG=$(printf 'r%s' "$RATIO" | tr '.' 'p')
CKPT=$ROOT/checkpoints/fm_time_moe_${TAG}
VAL_OUT=$ROOT/results/fm_time_moe_${TAG}_val
TEST_OUT=$ROOT/results/fm_time_moe_${TAG}_test
cd "$ROOT"
"$FM_PY" "$HERE/adapter.py" --mode fit --ratio "$RATIO" \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT"
"$FM_PY" "$HERE/adapter.py" --mode predict \
  --split-file setting/TEP_IDV13_XMEAS07_val.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$VAL_OUT"
"$FM_PY" "$HERE/adapter.py" --mode predict \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$TEST_OUT"
