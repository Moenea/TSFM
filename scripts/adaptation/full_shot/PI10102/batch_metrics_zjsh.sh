#!/usr/bin/env bash
# Compute window-level (Category C) metrics for ALL ZJSH 2103 PI10102
# transition models in a single run: 6 Timer-XL variants (zero-shot S/MS,
# full-shot S/MS, partial15 S/MS) and 13 MyTimeXer-style baselines
# (CNNLSTM / DiPCALSTM / LSTMGRU / STAConvBiLSTM / TCNTransformer / TimeXer /
# GTProger / GTProgerV13).
#
# The shared YAML gives each model its own seq_len / pred_len, then
# `align_eval_to: {seq_len: 768, pred_len: 96}` filters every model's windows
# down to the same time slice — i.e. baselines (sl=30) are restricted to the
# region a Timer-XL window can also see, so all metrics are computed over the
# same evaluation timeline.
cd /home/aicode/sherwin/TSFM

python -u ./utils/batch_metrics.py \
  --config ./setting/batch_metrics_zjsh_pi10102ts.yaml
