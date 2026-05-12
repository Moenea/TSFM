#!/usr/bin/env bash
# Window-level (Category C) metrics for FIRST-ORDER DIFF models after their
# pred.npy / true.npy have been restored to raw PI10102 absolute values.
#
# Sister script: batch_metrics_zjsh.sh (raw signal version).
cd /home/aicode/sherwin/TSFM

python -u ./utils/batch_metrics.py \
  --config ./setting/batch_metrics_zjsh_pi10102ts_diff.yaml
