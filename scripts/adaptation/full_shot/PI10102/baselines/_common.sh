#!/usr/bin/env bash
# Shared environment for MyTimeXer baseline scripts on ZJSH PI10102 transitions.
# Sourced by every per-model launcher in this directory.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

ROOT_PATH=/home/aicode/sherwin/dataset/ZJSH/
SPLIT_FILE=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml

# Short-window setting from MyTimeXer scripts/forecast_exogenous/JJ_PC302C/splits.yaml.
SEQ_LEN=30
LABEL_LEN=15
PRED_LEN=15
PATCH_LEN=10
DROPOUT=0.2
BATCH_SIZE=128
NUM_WORKERS=2
TRAIN_EPOCHS=50
PATIENCE=5
LR=1e-3

# PI10102 transitions: 16 input variables, target = PI10102 (last column).
ENC_IN_MS=16
DEC_IN_MS=16
C_OUT_MS=1

ENC_IN_S=1
DEC_IN_S=1
C_OUT_S=1

cd /home/aicode/sherwin/TSFM
