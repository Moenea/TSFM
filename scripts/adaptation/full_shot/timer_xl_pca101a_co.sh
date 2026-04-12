#!/usr/bin/env bash
# Full-shot fine-tuning of Timer-XL on JJ PCA101A — Co mode (multivariate -> univariate via covariate mask).
#
# Loads all 6 columns; the model uses TimerCovariateMask so the target column
# (last column, 0202B_PCA101A) attends to all historical covariates while the
# covariates themselves attend only to their own history. Loss/metrics are
# extracted from the last channel only — see exp/exp_forecast.py:88-94, 160-166.
#
# Sister scripts:
#   timer_xl_pca101a_m.sh    — multivariate -> multivariate
#   timer_xl_pca101a_u.sh    — univariate target only

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=16          # 16 * 96 = 1536
token_len=96
seq_len=$[$token_num*$token_len]

root_path=/home/aicode/sherwin/dataset/JJ/
split_file=/home/aicode/sherwin/TSFM/setting/PCA101A.yaml
pretrain_ckpt=/home/aicode/sherwin/TSFM/checkpoint.pth

cd /home/aicode/sherwin/TSFM

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path splits.yaml \
  --split_file $split_file \
  --model_id PCA101A_Co_full_shot \
  --model $model_name \
  --data MultivariateDatasetYAMLSplit \
  --features MS \
  --covariate \
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
