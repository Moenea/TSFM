#!/usr/bin/env bash
set -euo pipefail
# Orchestrate the XMEAS10 few-shot curve for the TSFM side + gate + eval.
# PREREQUISITES (run these ONCE first, see WORKFLOW.md):
#   bash scripts/adaptation/xmeas10/run_fm_zeroshot.sh     # Time-MoE + Sundial (once)
#   bash scripts/adaptation/xmeas10/run_baselines.sh       # 5 baselines (long, ~35 runs)
# Then this script: per ratio, train Timer-XL raw/diff + fuse gate + evaluate all; then aggregate.
HERE=$(cd "$(dirname "$0")" && pwd)
RATIOS=${RATIOS:-"0.01 0.02 0.05 0.1 0.25 0.5 1.0"}
for r in $RATIOS; do
  echo "############ XMEAS10 ratio=$r ############"
  SUBSET_RATIO=$r bash "$HERE/run_timerxl.sh"
  SUBSET_RATIO=$r bash "$HERE/run_gate_eval.sh"
done
/home/aicode/miniconda3/envs/tsfm/bin/python "$HERE/collect_compare.py"
echo "ALL DONE -> results/TEP_IDV13_XMEAS10_Summary/comparison.{csv,png} + figures/few-shot/TEP/IDV13_XMEAS10/"
