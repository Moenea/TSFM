#!/usr/bin/env bash
# Full-shot fine-tuning of Timer-XL on ZJSH 2103 PI10102 transition episodes
# — MS mode (multivariate history -> univariate target).
#
# Loads all 16 variables as input context; the TimerCovariateMask + --last_token
# flag constrain the supervised output to ONLY the last column (PI10102,
# guaranteed to be the target by the export script). Each sample still has
# shape [B, L, 16] going in; loss is computed on the final channel only.
#
# Sister script:
#   timer_xl_zjsh_pi10102ts_s.sh  — S mode (PI10102 alone, self-predicts)

export CUDA_VISIBLE_DEVICES=1

model_name=timer_xl
token_num=8
token_len=96
seq_len=$[$token_num*$token_len]

root_path=/home/aicode/sherwin/dataset/ZJSH/
split_file=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml
pretrain_ckpt=/home/aicode/sherwin/TSFM/checkpoint.pth

cd /home/aicode/sherwin/TSFM

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path splits.yaml \
  --split_file $split_file \
  --model_id ZJSH_PI10102TS_MS_full_shot \
  --model $model_name \
  --data MultivariateDatasetYAMLSplit \
  --features MS \
  --covariate \
  --last_token \
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
