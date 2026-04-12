#!/usr/bin/env bash
# Inference for all 5 Timer-XL models on PCA101A test set.
# Results (pred.npy, true.npy, metrics.npy) saved to ./results/{test_dir}/
#
# Models:
#   1. Zero-shot U   — pretrained checkpoint, univariate target only
#   2. Zero-shot M   — pretrained checkpoint, multivariate (all 6 channels)
#   3. Zero-shot Co  — pretrained checkpoint, covariate mode (M→1)
#   4. Fine-tuned U  — full-shot fine-tuned, univariate
#   5. Fine-tuned Co — full-shot fine-tuned, covariate mode

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=16
token_len=96
seq_len=$[$token_num*$token_len]

root_path=/home/aicode/sherwin/dataset/JJ/
split_file=/home/aicode/sherwin/TSFM/setting/PCA101A.yaml

cd /home/aicode/sherwin/TSFM

# --- shared args (everything except features/covariate/test_dir) ---
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
echo "1/5  Zero-shot U"
echo "============================================"
python -u run.py $COMMON \
  --model_id PCA101A_zeroshot_U \
  --features S \
  --test_dir zeroshot_PCA101A_U

echo "============================================"
echo "2/5  Zero-shot M"
echo "============================================"
python -u run.py $COMMON \
  --model_id PCA101A_zeroshot_M \
  --features M \
  --test_dir zeroshot_PCA101A_M

echo "============================================"
echo "3/5  Zero-shot Co"
echo "============================================"
python -u run.py $COMMON \
  --model_id PCA101A_zeroshot_Co \
  --features MS \
  --covariate \
  --test_dir zeroshot_PCA101A_Co