#!/usr/bin/env bash
# Compute window-level (Category C) metrics for Timer-XL models fine-tuned
# on ZJSH 2103 PI10102 transition episodes. Covers S and MS runs.
cd /home/aicode/sherwin/TSFM

python -u ./utils/batch_metrics.py \
  --config ./setting/batch_metrics_zjsh_pi10102ts.yaml
