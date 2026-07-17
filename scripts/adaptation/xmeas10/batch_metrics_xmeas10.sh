#!/usr/bin/env bash
# Window-level (Category C) prognosis metrics for the XMEAS10 few-shot study:
# recall rate (ratio_pred_in_true_alarm_patches), false prognosis rate
# (ratio_pred_in_no_true_alarm_patches) and mean lead time
# (mean_lead_time_patch), plus clean / quality-filtered variants — for all
# 15 methods x 7 few-shot ratios, using mu+-3sigma limits from Run1-7
# pre-onset data (setting/limits_tep_xmeas10.csv).
#
# Per-ratio outputs:
#   results/"XMEAS10 Purge Rate_Summary_{ratio}"/summary.csv (+ .md)
#   <each result dir>/metrics_C.json
# Cross-ratio aggregate (collect_batch_metrics.py):
#   results/TEP_IDV13_XMEAS10_Summary/batch_metrics_by_ratio.csv
#   figures/few-shot/TEP/IDV13_XMEAS10/batch_metrics_curves.png
set -euo pipefail
cd /home/aicode/sherwin/TSFM
PY=/home/aicode/miniconda3/envs/tsfm/bin/python

$PY scripts/adaptation/xmeas10/gen_batch_metrics_cfgs.py

for r in r0p01 r0p02 r0p05 r0p1 r0p25 r0p5 r1p0; do
  echo "===== batch_metrics $r ====="
  $PY -u utils/batch_metrics.py \
    --config "setting/batch_metrics_tep_xmeas10_${r}.yaml" \
    --summary-suffix "_${r}" \
    --figure-suffix "_${r}"
done

$PY scripts/adaptation/xmeas10/collect_batch_metrics.py
