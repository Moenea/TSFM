#!/usr/bin/env bash
# Zero-shot inference for Timer-XL on ZJSH 2103 PI10102 transition episodes.
# Two configurations:
#   1. Zero-shot S   — pretrained checkpoint, univariate target only
#   2. Zero-shot MS  — pretrained checkpoint, covariate mode (16 vars -> PI10102)
#
# When --is_training 0, run.py calls exp.test(setting, test=1), which
# HARD-CODES the checkpoint path to ./checkpoints/{test_dir}/checkpoint.pth
# (see exp/exp_forecast.py:268-281). It does NOT honour --pretrain_model_path
# in that branch. So we symlink the project-root Timer-XL checkpoint into the
# expected location before running — same pattern used for zeroshot_PCA101A_*.
#
# Outputs:
#   ./results/zeroshot_ZJSH_PI10102TS_S/{pred,true,metrics}.npy
#   ./results/zeroshot_ZJSH_PI10102TS_MS/{pred,true,metrics}.npy
# These are consumed by setting/batch_metrics_zjsh_pi10102ts.yaml.

export CUDA_VISIBLE_DEVICES=1

model_name=timer_xl
token_num=8           # 8 * 96 = 768, matches the full-shot fine-tunes
token_len=96
seq_len=$[$token_num*$token_len]

root_path=/home/aicode/sherwin/dataset/ZJSH/
split_file=/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml
pretrain_ckpt=/home/aicode/sherwin/TSFM/checkpoint.pth

cd /home/aicode/sherwin/TSFM

# Stage the pretrain checkpoint where exp.test() expects to find it.
for d in zeroshot_ZJSH_PI10102TS_S zeroshot_ZJSH_PI10102TS_MS; do
  mkdir -p "./checkpoints/$d"
  ln -sf "$pretrain_ckpt" "./checkpoints/$d/checkpoint.pth"
done

# --- shared args (everything except features/covariate/test_dir/model_id) ---
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
"

echo "============================================"
echo "1/2  Zero-shot S  (PI10102 alone)"
echo "============================================"
python -u run.py $COMMON \
  --model_id ZJSH_PI10102TS_zeroshot_S \
  --features S \
  --test_dir zeroshot_ZJSH_PI10102TS_S

echo "============================================"
echo "2/2  Zero-shot MS  (16 vars -> PI10102)"
echo "============================================"
python -u run.py $COMMON \
  --model_id ZJSH_PI10102TS_zeroshot_MS \
  --features MS \
  --covariate \
  --last_token \
  --test_dir zeroshot_ZJSH_PI10102TS_MS
