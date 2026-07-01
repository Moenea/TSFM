#!/usr/bin/env bash
set -euo pipefail
# Sundial expert: zero-shot generative TSFM.
# Generates val (Run8) and test (Run9-10) predictions if not already present.
HERE=$(cd "$(dirname "$0")" && pwd)
TSFM_PY=${TSFM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"

VAL_OUT=$ROOT/results/fm_sundial_zeroshot_val
TEST_OUT=$ROOT/results/fm_sundial_zeroshot_test

cd "$ROOT"

if [ ! -d "$TEST_OUT" ]; then
  echo "[sundial/run.sh] Generating zero-shot Sundial test predictions..."
  CUDA_VISIBLE_DEVICES=0 "$TSFM_PY" "$HERE/adapter.py" \
    --mode predict --zero-shot \
    --num-samples 20 --batch-size 32 \
    --split-file setting/TEP_IDV13_XMEAS07.yaml \
    --data-root "$DATA_ROOT" --target "$TARGET" \
    --out-dir "$TEST_OUT"
else
  echo "[sundial/run.sh] Test predictions already exist, skipping."
fi

if [ ! -d "$VAL_OUT" ]; then
  echo "[sundial/run.sh] Generating zero-shot Sundial val predictions..."
  CUDA_VISIBLE_DEVICES=0 "$TSFM_PY" "$HERE/adapter.py" \
    --mode predict --zero-shot \
    --num-samples 20 --batch-size 32 \
    --split-file setting/TEP_IDV13_XMEAS07_val.yaml \
    --data-root "$DATA_ROOT" --target "$TARGET" \
    --out-dir "$VAL_OUT"
else
  echo "[sundial/run.sh] Val predictions already exist, skipping."
fi
