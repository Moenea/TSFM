#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"$PYTHON_BIN" "$(dirname "$0")/prepare_diff.py"

"$PYTHON_BIN" -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --data_path splits.yaml \
  --split_file "$DIFF_SPLIT" \
  --target "$TARGET" \
  --model_id "$DIFF_MODEL_ID" \
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
  --restore_diff_to_raw \
  --raw_split_file "$RAW_SPLIT" \
  --restore_target "$TARGET" \
  --adaptation \
  --pretrain_model_path "$PRETRAIN_CKPT"
