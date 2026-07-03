#!/usr/bin/env bash
set -euo pipefail
# ===================================================================
# XMEAS10 (Purge flow) few-shot study — shared config.
#
# Task: predict XMEAS10 (Purge Rate) at horizon 15, few-shot on TEP IDV13.
# Splits are IDENTICAL to the XMEAS07 study (Run1-7 train / Run8 val / Run9-10 test,
# seed-2021 subset, seq_len=96, output_token_len=96 -> 1810 windows/file).
#
# Per-method inputs (see WORKFLOW.md):
#   Timer-XL raw/diff : MS, 5 variables {XMEAS10,15,17,XMV06,XMV11} -> XMEAS10  (this file)
#   Time-MoE/Sundial  : univariate XMEAS10 -> XMEAS10   (run_fm_zeroshot.sh)
#   Baselines         : MS-diff, same 5 variables -> XMEAS10   (run_baselines.sh)
#   Gate              : fuse experts' XMEAS10 predictions   (run_gate_eval.sh)
# ===================================================================
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
PYTHON_BIN=${PYTHON_BIN:-/home/aicode/miniconda3/envs/tsfm/bin/python}
PRETRAIN_CKPT=$ROOT/checkpoint.pth
TARGET="XMEAS10 Purge Rate"

# 5-variable MS splits (XMEAS10 last). raw + first-difference.
RAW_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS10_5var.yaml
DIFF_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS10_5var_diff.yaml
RAW_VAL_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS10_5var_val.yaml
DIFF_VAL_SPLIT=$ROOT/setting/TEP_IDV13_XMEAS10_5var_diff_val.yaml

SEQ_LEN=96
TOKEN_LEN=96
PRED_LEN=96
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-10}
GPU_PHYSICAL=${GPU_PHYSICAL:-0}
ENC_IN=5          # 5 input variables (MS)

SUBSET_RATIO=${SUBSET_RATIO:-1.0}
export SUBSET_RATIO
RATIO_TAG=$(printf 'r%s' "$SUBSET_RATIO" | tr '.' 'p')
DATA_NAME=MultivariateDatasetYAMLSplitFewShot

RAW_MODEL_ID=TEP_IDV13_XMEAS10_5var_raw_few_${RATIO_TAG}
DIFF_MODEL_ID=TEP_IDV13_XMEAS10_5var_diff_few_${RATIO_TAG}
RAW_VAL_MODEL_ID=TEP_IDV13_XMEAS10_5var_raw_val_few_${RATIO_TAG}
DIFF_VAL_MODEL_ID=TEP_IDV13_XMEAS10_5var_diff_val_few_${RATIO_TAG}

SUF=timer_xl_${DATA_NAME}_sl${SEQ_LEN}_it${TOKEN_LEN}_ot${TOKEN_LEN}_lr5e-06_bt${BATCH_SIZE}_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAW_SETTING=forecast_${RAW_MODEL_ID}_${SUF}
DIFF_SETTING=forecast_${DIFF_MODEL_ID}_${SUF}
RAW_VAL_SETTING=forecast_${RAW_VAL_MODEL_ID}_${SUF}
DIFF_VAL_SETTING=forecast_${DIFF_VAL_MODEL_ID}_${SUF}

export CUDA_VISIBLE_DEVICES=$GPU_PHYSICAL
cd "$ROOT"

# Timer-XL adaptation training, MS 5-var -> XMEAS10 (covariate=last channel).
# $1 model_id, $2 split_file, then extra passthrough args (diff-restore flags).
timer_args_ms() {
  local model_id=$1; local split_file=$2; shift 2
  "$PYTHON_BIN" -u run.py \
    --task_name forecast --is_training 1 \
    --root_path "$DATA_ROOT" --data_path splits.yaml --split_file "$split_file" \
    --target "$TARGET" --model_id "$model_id" --model timer_xl --data "$DATA_NAME" \
    --subset_rand_ratio "$SUBSET_RATIO" \
    --features MS --covariate --last_token \
    --enc_in "$ENC_IN" --dec_in "$ENC_IN" --c_out 1 \
    --seq_len "$SEQ_LEN" --input_token_len "$TOKEN_LEN" --output_token_len "$TOKEN_LEN" \
    --test_seq_len "$SEQ_LEN" --test_pred_len "$PRED_LEN" \
    --e_layers 8 --d_model 1024 --d_ff 2048 --n_heads 8 \
    --batch_size "$BATCH_SIZE" --num_workers 4 --learning_rate 5e-6 \
    --train_epochs "$EPOCHS" --patience 3 --gpu 0 --cosine --tmax "$EPOCHS" \
    --use_norm --valid_last --adaptation --pretrain_model_path "$PRETRAIN_CKPT" \
    "$@"
}

# Timer-XL inference from a trained checkpoint (val split), MS 5-var.
# $1 model_id, $2 split_file, $3 train_setting(dir), $4 result_setting, then passthrough.
inference_args_ms() {
  local model_id=$1; local split_file=$2; local train_setting=$3; local result_setting=$4; shift 4
  "$PYTHON_BIN" -u run.py \
    --task_name forecast --is_training 0 \
    --root_path "$DATA_ROOT" --data_path splits.yaml --split_file "$split_file" \
    --target "$TARGET" --model_id "$model_id" --model timer_xl --data "$DATA_NAME" \
    --subset_rand_ratio "$SUBSET_RATIO" \
    --features MS --covariate --last_token \
    --enc_in "$ENC_IN" --dec_in "$ENC_IN" --c_out 1 \
    --seq_len "$SEQ_LEN" --input_token_len "$TOKEN_LEN" --output_token_len "$TOKEN_LEN" \
    --test_seq_len "$SEQ_LEN" --test_pred_len "$PRED_LEN" \
    --e_layers 8 --d_model 1024 --d_ff 2048 --n_heads 8 \
    --batch_size "$BATCH_SIZE" --num_workers 4 --learning_rate 5e-6 \
    --gpu 0 --cosine --use_norm \
    --test_dir "$train_setting" --test_file_name checkpoint.pth --result_setting "$result_setting" \
    "$@"
}
