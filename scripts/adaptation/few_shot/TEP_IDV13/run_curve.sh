#!/usr/bin/env bash
set -euo pipefail
# Sweep the few-shot fraction and build the "training-sample-count -> performance"
# curve. Override the ratio list with RATIOS="...".
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/_common.sh"   # ROOT, PYTHON_BIN, GPU, ...

RATIOS=${RATIOS:-"0.01 0.02 0.05 0.1 0.25 0.5 1.0"}
SUMMARY_DIR=$ROOT/results/TEP_IDV13_XMEAS07_FewShot_Summary
LOG_DIR=$SUMMARY_DIR/logs
mkdir -p "$LOG_DIR"

for r in $RATIOS; do
  tag=$(printf 'r%s' "$r" | tr '.' 'p')
  echo "############ FEW-SHOT ratio=$r (tag=$tag) ############"
  SUBSET_RATIO=$r bash "$HERE/run_all.sh" 2>&1 | tee "$LOG_DIR/few_${tag}.log"
done

"$PYTHON_BIN" "$HERE/collect_curve.py" \
  --summary-dir "$SUMMARY_DIR" \
  --log-dir "$LOG_DIR" \
  --ratios $RATIOS \
  --out-csv "$SUMMARY_DIR/curve.csv" \
  --out-json "$SUMMARY_DIR/curve.json"
