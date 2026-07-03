#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
PYTHON_BIN=${PYTHON_BIN:-/home/aicode/miniconda3/envs/tsfm/bin/python}
PRETRAIN_CKPT=$ROOT/checkpoint.pth
RAW_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS07.yaml
DIFF_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS07_diff.yaml
RAW_VAL_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS07_val.yaml
DIFF_VAL_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS07_diff_val.yaml
TARGET="XMEAS07 Reactor Pressure"

SEQ_LEN=96
TOKEN_LEN=96
PRED_LEN=96
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-10}
GPU_PHYSICAL=${GPU_PHYSICAL:-1}

RAW_MODEL_ID=TEP_IDV13_XMEAS07_S_full_shot
DIFF_MODEL_ID=TEP_IDV13_XMEAS07_S_full_shot_DIFF
RAW_VAL_MODEL_ID=TEP_IDV13_XMEAS07_S_val
DIFF_VAL_MODEL_ID=TEP_IDV13_XMEAS07_S_DIFF_val

RAW_SETTING=forecast_${RAW_MODEL_ID}_timer_xl_MultivariateDatasetYAMLSplit_sl${SEQ_LEN}_it${TOKEN_LEN}_ot${TOKEN_LEN}_lr5e-06_bt${BATCH_SIZE}_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
DIFF_SETTING=forecast_${DIFF_MODEL_ID}_timer_xl_MultivariateDatasetYAMLSplit_sl${SEQ_LEN}_it${TOKEN_LEN}_ot${TOKEN_LEN}_lr5e-06_bt${BATCH_SIZE}_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAW_VAL_SETTING=forecast_${RAW_VAL_MODEL_ID}_timer_xl_MultivariateDatasetYAMLSplit_sl${SEQ_LEN}_it${TOKEN_LEN}_ot${TOKEN_LEN}_lr5e-06_bt${BATCH_SIZE}_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
DIFF_VAL_SETTING=forecast_${DIFF_VAL_MODEL_ID}_timer_xl_MultivariateDatasetYAMLSplit_sl${SEQ_LEN}_it${TOKEN_LEN}_ot${TOKEN_LEN}_lr5e-06_bt${BATCH_SIZE}_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0

export CUDA_VISIBLE_DEVICES=$GPU_PHYSICAL
cd "$ROOT"

timer_args() {
  local model_id=$1
  local split_file=$2
  "$PYTHON_BIN" -u run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path "$DATA_ROOT" \
    --data_path splits.yaml \
    --split_file "$split_file" \
    --target "$TARGET" \
    --model_id "$model_id" \
    --model timer_xl \
    --data MultivariateDatasetYAMLSplit \
    --features S \
    --seq_len "$SEQ_LEN" \
    --input_token_len "$TOKEN_LEN" \
    --output_token_len "$TOKEN_LEN" \
    --test_seq_len "$SEQ_LEN" \
    --test_pred_len "$PRED_LEN" \
    --e_layers 8 \
    --d_model 1024 \
    --d_ff 2048 \
    --n_heads 8 \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --learning_rate 5e-6 \
    --train_epochs "$EPOCHS" \
    --patience 3 \
    --gpu 0 \
    --cosine \
    --tmax "$EPOCHS" \
    --use_norm \
    --valid_last \
    --adaptation \
    --pretrain_model_path "$PRETRAIN_CKPT"
}

inference_args() {
  local model_id=$1
  local split_file=$2
  local train_setting=$3
  local result_setting=$4
  shift 4
  "$PYTHON_BIN" -u run.py \
    --task_name forecast \
    --is_training 0 \
    --root_path "$DATA_ROOT" \
    --data_path splits.yaml \
    --split_file "$split_file" \
    --target "$TARGET" \
    --model_id "$model_id" \
    --model timer_xl \
    --data MultivariateDatasetYAMLSplit \
    --features S \
    --seq_len "$SEQ_LEN" \
    --input_token_len "$TOKEN_LEN" \
    --output_token_len "$TOKEN_LEN" \
    --test_seq_len "$SEQ_LEN" \
    --test_pred_len "$PRED_LEN" \
    --e_layers 8 \
    --d_model 1024 \
    --d_ff 2048 \
    --n_heads 8 \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --learning_rate 5e-6 \
    --gpu 0 \
    --cosine \
    --use_norm \
    --test_dir "$train_setting" \
    --test_file_name checkpoint.pth \
    --result_setting "$result_setting" \
    "$@"
}
