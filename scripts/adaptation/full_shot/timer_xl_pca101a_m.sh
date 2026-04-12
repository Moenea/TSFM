#!/usr/bin/env bash
# Full-shot fine-tuning of Timer-XL on JJ PCA101A — M mode (multivariate -> multivariate).
#
# All 6 columns are loaded; the model jointly forecasts all 6 channels with
# full inter-channel attention (TimerMultivariateMask). To switch task type
# without rewriting this script, see:
#   timer_xl_pca101a_co.sh   — covariate mode (M -> univariate target)
#   timer_xl_pca101a_u.sh    — univariate mode (target column only, no covariates)
#
# All three scripts share the SAME setting/PCA101A.yaml split file.
#
# Pretrained checkpoint: /home/aicode/sherwin/TSFM/checkpoint.pth. The
# architecture below MUST match it: e_layers=8 d_model=1024 d_ff=2048 n_heads=8
# input_token_len=96 output_token_len=96.

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=16          # 16 * 96 = 1536, matches load_pth_ckpt.ipynb's lookback
token_len=96
seq_len=$[$token_num*$token_len]

# Where the relative paths inside the YAML splits file are resolved from.
# PCA101A.yaml lists files as "PCA101A/2FCC_PCA101A_smooth_*.csv", so root must
# be the parent JJ/ directory.
root_path=/home/aicode/sherwin/dataset/JJ/

# TSFM-owned YAML split file (one file serves M / Co / U modes — see comments
# inside the YAML).
split_file=/home/aicode/sherwin/TSFM/setting/PCA101A.yaml

# Pretrained checkpoint to fine-tune from
pretrain_ckpt=/home/aicode/sherwin/TSFM/checkpoint.pth

cd /home/aicode/sherwin/TSFM

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path splits.yaml \
  --split_file $split_file \
  --model_id PCA101A_M_full_shot \
  --model $model_name \
  --data MultivariateDatasetYAMLSplit \
  --features M \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
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
  --adaptation \
  --pretrain_model_path $pretrain_ckpt
