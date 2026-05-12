#!/usr/bin/env bash
# Patch-count ablation for Timer-XL on ZJSH PI10102 DIFF→ABS, MS mode.
# Runs patch_num in PATCH_NUMS (default: 1 3 5). Names include _p${patch_num}
# and seq_len changes, so checkpoints/results never overwrite the 8-patch run.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-/home/aicode/miniconda3/envs/tsfm/bin/python}
PATCH_NUMS=${PATCH_NUMS:-"1 3 5"}

model_name=timer_xl
token_len=96
root_path=/home/aicode/sherwin/dataset/ZJSH/
split_file=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts_diff.yaml
raw_split_file=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml
pretrain_ckpt=/home/aicode/sherwin/TSFM/checkpoint.pth

cd /home/aicode/sherwin/TSFM

for token_num in $PATCH_NUMS; do
  seq_len=$((token_num * token_len))
  echo "============================================"
  echo "Timer-XL MS DIFF patch ablation: p${token_num}, seq_len=${seq_len}"
  echo "============================================"
  "$PYTHON_BIN" -u run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path splits.yaml \
    --split_file "$split_file" \
    --model_id "ZJSH_PI10102TS_MS_full_shot_DIFF_p${token_num}" \
    --model "$model_name" \
    --data MultivariateDatasetYAMLSplit \
    --features MS \
    --covariate \
    --last_token \
    --seq_len "$seq_len" \
    --input_token_len "$token_len" \
    --output_token_len "$token_len" \
    --test_seq_len "$seq_len" \
    --test_pred_len 96 \
    --e_layers 8 \
    --d_model 1024 \
    --d_ff 2048 \
    --n_heads 8 \
    --batch_size 32 \
    --num_workers 4 \
    --learning_rate 5e-6 \
    --train_epochs 10 \
    --patience 3 \
    --gpu 0 \
    --cosine \
    --tmax 10 \
    --use_norm \
    --valid_last \
    --restore_diff_to_raw \
    --raw_split_file "$raw_split_file" \
    --restore_target PI10102 \
    --adaptation \
    --pretrain_model_path "$pretrain_ckpt"
done
