#!/usr/bin/env bash
# Zero-shot inference for Timer-XL on FIRST-ORDER-DIFF ZJSH PI10102 transitions.
#
# The model predicts differenced PI10102. run.py stores those arrays as
# pred_diff.npy / true_diff.npy, then integrates them back to raw PI10102 and
# writes the raw-scale arrays to pred.npy / true.npy.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

model_name=timer_xl
token_num=8
token_len=96
seq_len=$[$token_num*$token_len]

root_path=/home/aicode/sherwin/dataset/ZJSH/
split_file=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts_diff.yaml
raw_split_file=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml
pretrain_ckpt=/home/aicode/sherwin/TSFM/checkpoint.pth

cd /home/aicode/sherwin/TSFM

for d in zeroshot_ZJSH_PI10102TS_S_DIFF zeroshot_ZJSH_PI10102TS_MS_DIFF; do
  mkdir -p "./checkpoints/$d"
  ln -sf "$pretrain_ckpt" "./checkpoints/$d/checkpoint.pth"
done

COMMON="
  --task_name forecast
  --is_training 0
  --root_path $root_path
  --data_path splits.yaml
  --split_file $split_file
  --model $model_name
  --data MultivariateDatasetYAMLSplit
  --seq_len $seq_len
  --input_token_len $token_len
  --output_token_len $token_len
  --test_seq_len $seq_len
  --test_pred_len 96
  --e_layers 8
  --d_model 1024
  --d_ff 2048
  --n_heads 8
  --batch_size 32
  --num_workers 4
  --learning_rate 5e-6
  --gpu 0
  --cosine
  --tmax 10
  --use_norm
  --restore_diff_to_raw
  --raw_split_file $raw_split_file
  --restore_target PI10102
"

echo "============================================"
echo "1/2  Zero-shot DIFF S  (PI10102 diff alone)"
echo "============================================"
python -u run.py $COMMON \
  --model_id ZJSH_PI10102TS_zeroshot_S_DIFF \
  --features S \
  --test_dir zeroshot_ZJSH_PI10102TS_S_DIFF

echo "============================================"
echo "2/2  Zero-shot DIFF MS  (16 diff vars -> PI10102 diff)"
echo "============================================"
python -u run.py $COMMON \
  --model_id ZJSH_PI10102TS_zeroshot_MS_DIFF \
  --features MS \
  --covariate \
  --last_token \
  --test_dir zeroshot_ZJSH_PI10102TS_MS_DIFF
