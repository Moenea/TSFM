#!/usr/bin/env bash
# Full-shot fine-tuning of Timer-XL on ZJSH 2103 PI10102 transition episodes
# — S mode (univariate target only).
#
# Loads ONLY the target column (PI10102, declared in ZJSH_PI10102ts.yaml as
# the target AND guaranteed by the export script to be the LAST column of
# every transition CSV). Each sample shape becomes [B, L, 1]; the model
# self-predicts using PI10102's own history with no covariates.
#
# Sister script:
#   timer_xl_zjsh_pi10102ts_ms.sh  — MS mode (multivariate history -> PI10102)
#
# seq_len=768 (8 * 96) chosen because the shortest transition is 904 samples;
# every transition produces >= 41 sliding windows at pred_len=96.

export CUDA_VISIBLE_DEVICES=1

model_name=timer_xl
token_num=8           # 8 * 96 = 768
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
  --model_id ZJSH_PI10102TS_S_full_shot \
  --model $model_name \
  --data MultivariateDatasetYAMLSplit \
  --features S \
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
