#!/usr/bin/env bash
# run_poc.sh — ratio=1.0 proof-of-concept orchestrator for the N-expert FM gate.
# Wires all available experts over Timer-XL diff/raw, appending Time-MoE and
# Sundial only when their adapters and result dirs exist. MOIRAI is dropped
# (uni2ts incompatible with tsfm env). Always run with CUDA_VISIBLE_DEVICES=0.
#
# Usage:
#   bash scripts/adaptation/foundation_experts/run_poc.sh
#
# Env vars (all optional, defaults shown):
#   TSFM_PY   — Python interpreter for the gate/eval scripts (default: tsfm env)
#   FM_PY     — Python interpreter for the FM adapter scripts (default: TSFM_PY)
#   TAG       — result-tag suffix (default: r1p0)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
TSFM_PY=${TSFM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
FM_PY=${FM_PY:-$TSFM_PY}
export FM_PY TSFM_PY
TAG=${TAG:-r1p0}
cd "$ROOT"

# ---------------------------------------------------------------------------
# 0) Timer-XL diff/raw at ratio=1.0 — skip if already present
# ---------------------------------------------------------------------------
DIFF=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_DIFF_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAW=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
DIFF_VAL=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_DIFF_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAW_VAL=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0

if [ ! -d "$DIFF" ]; then
  echo "[run_poc] Timer-XL r1p0 DIFF dir absent — running run_curve.sh..."
  RATIOS="1.0" bash scripts/adaptation/few_shot/TEP_IDV13/run_curve.sh
else
  echo "[run_poc] Timer-XL r1p0 dirs present — skipping re-train."
fi

# ---------------------------------------------------------------------------
# 1) New-model adapters — each guarded by adapter existence
#    MOIRAI is intentionally excluded (uni2ts incompatibility).
# ---------------------------------------------------------------------------
# Time-MoE is always zero-shot — use the same constant dirs as run_curve_multi.sh.
if [ -f "$HERE/time_moe/adapter.py" ]; then
  if [ ! -d "results/fm_time_moe_zeroshot_test" ]; then
    echo "[run_poc] Generating zero-shot Time-MoE test predictions..."
    CUDA_VISIBLE_DEVICES=0 "$TSFM_PY" "$HERE/time_moe/adapter.py" \
      --mode predict --zero-shot \
      --split-file setting/TEP_IDV13_XMEAS07.yaml \
      --data-root "$DATA_ROOT" --target "$TARGET" \
      --horizon 15 --out-dir results/fm_time_moe_zeroshot_test --device cuda:0
  else
    echo "[run_poc] results/fm_time_moe_zeroshot_test already exists — skipping."
  fi
  if [ ! -d "results/fm_time_moe_zeroshot_val" ]; then
    echo "[run_poc] Generating zero-shot Time-MoE val predictions..."
    CUDA_VISIBLE_DEVICES=0 "$TSFM_PY" "$HERE/time_moe/adapter.py" \
      --mode predict --zero-shot \
      --split-file setting/TEP_IDV13_XMEAS07_val.yaml \
      --data-root "$DATA_ROOT" --target "$TARGET" \
      --horizon 15 --out-dir results/fm_time_moe_zeroshot_val --device cuda:0
  else
    echo "[run_poc] results/fm_time_moe_zeroshot_val already exists — skipping."
  fi
else
  echo "[run_poc] time_moe/adapter.py not found — skipping Time-MoE."
fi

if [ -f "$HERE/sundial/run.sh" ] && [ ! -d "results/fm_sundial_${TAG}_test" ]; then
  echo "[run_poc] Running Sundial adapter..."
  SUBSET_RATIO=1.0 CUDA_VISIBLE_DEVICES=0 FM_PY=$FM_PY \
    bash "$HERE/sundial/run.sh"
elif [ -f "$HERE/sundial/run.sh" ]; then
  echo "[run_poc] results/fm_sundial_${TAG}_test already exists — skipping Sundial."
else
  echo "[run_poc] sundial/run.sh not found — skipping Sundial (weights not ready)."
fi

# ---------------------------------------------------------------------------
# 2) Build expert lists dynamically
#    diff and raw are always included (Timer-XL base).
#    time_moe and sundial included only if their result dirs exist.
# ---------------------------------------------------------------------------
GATE=results/ensemble_Gate-multi_${TAG}_test

EXPERTS=(--expert "diff:$DIFF" --expert "raw:$RAW")
VAL_EXPERTS=(--val-expert "diff:$DIFF_VAL" --val-expert "raw:$RAW_VAL")

if [ -d "results/fm_time_moe_zeroshot_test" ]; then
  EXPERTS+=(--expert "time_moe:results/fm_time_moe_zeroshot_test")
  VAL_EXPERTS+=(--val-expert "time_moe:results/fm_time_moe_zeroshot_val")
  echo "[run_poc] Time-MoE zeroshot dir found — including in gate."
else
  echo "[run_poc] results/fm_time_moe_zeroshot_test absent — excluding Time-MoE from gate."
fi

if [ -d "results/fm_sundial_${TAG}_test" ]; then
  EXPERTS+=(--expert "sundial:results/fm_sundial_${TAG}_test")
  VAL_EXPERTS+=(--val-expert "sundial:results/fm_sundial_${TAG}_val")
  echo "[run_poc] Sundial test dir found — including in gate."
else
  echo "[run_poc] results/fm_sundial_${TAG}_test absent — excluding Sundial from gate."
fi

echo "[run_poc] Fusing ${#EXPERTS[@]} expert entries..."
"$TSFM_PY" "$HERE/fuse_gate_multi.py" \
  "${EXPERTS[@]}" \
  "${VAL_EXPERTS[@]}" \
  --data-root "$DATA_ROOT" \
  --train-split setting/TEP_IDV13_XMEAS07.yaml \
  --val-split  setting/TEP_IDV13_XMEAS07_val.yaml \
  --test-split setting/TEP_IDV13_XMEAS07.yaml \
  --target "$TARGET" \
  --output-dir "$GATE"

# ---------------------------------------------------------------------------
# 3) Evaluate every expert + the gate
# ---------------------------------------------------------------------------
SUMMARY=results/TEP_IDV13_XMEAS07_FM_Summary
mkdir -p "$SUMMARY"

EVAL=(
  --expert "diff:$DIFF"
  --expert "raw:$RAW"
  --expert "gate:$GATE"
)
if [ -d "results/fm_time_moe_zeroshot_test" ]; then
  EVAL+=(--expert "time_moe:results/fm_time_moe_zeroshot_test")
fi
if [ -d "results/fm_sundial_${TAG}_test" ]; then
  EVAL+=(--expert "sundial:results/fm_sundial_${TAG}_test")
fi

"$TSFM_PY" "$HERE/evaluate_multi.py" \
  "${EVAL[@]}" \
  --data-root "$DATA_ROOT" \
  --split setting/TEP_IDV13_XMEAS07.yaml \
  --target "$TARGET" \
  --output "$SUMMARY/metrics_${TAG}.json"

echo "[run_poc] Done -> $SUMMARY/metrics_${TAG}.json"
