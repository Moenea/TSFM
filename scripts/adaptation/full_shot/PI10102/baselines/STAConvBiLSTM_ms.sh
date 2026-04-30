#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path "$ROOT_PATH" --split_file "$SPLIT_FILE" \
  --data MultivariateDatasetYAMLSplit \
  --model_id ZJSH_PI10102TS_STAConvBiLSTM_MS --model STAConvBiLSTM \
  --features MS --covariate --last_token \
  --seq_len $SEQ_LEN --input_token_len $PRED_LEN --output_token_len $PRED_LEN \
  --test_seq_len $SEQ_LEN --test_pred_len $PRED_LEN --nonautoregressive \
  --enc_in $ENC_IN_MS --dec_in $DEC_IN_MS --c_out $C_OUT_MS \
  --label_len $LABEL_LEN --pred_len $PRED_LEN \
  --d_model 268 --d_ff 2048 --e_layers 1 --dropout $DROPOUT \
  --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR \
  --train_epochs $TRAIN_EPOCHS --patience $PATIENCE --gpu 0 --use_norm \
  --des STAConvBiLSTM-MS
