#!/usr/bin/env bash
# Compute window-level (Category C) metrics for all Timer-XL PCA101A models.
cd /home/aicode/sherwin/TSFM

python -u ./utils/batch_metrics.py \
  --config ./setting/batch_metrics_pca101a.yaml
