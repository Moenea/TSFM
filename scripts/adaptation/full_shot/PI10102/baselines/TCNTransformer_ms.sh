#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

# TCNTransformer uses tail-aware MSE in the original MyTimeXer training recipe.
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path "$ROOT_PATH" --split_file "$SPLIT_FILE" \
  --data MultivariateDatasetYAMLSplit \
  --model_id ZJSH_PI10102TS_TCNTransformer_MS --model TCNTransformer \
  --features MS --covariate --last_token \
  --seq_len $SEQ_LEN --input_token_len $PRED_LEN --output_token_len $PRED_LEN \
  --test_seq_len $SEQ_LEN --test_pred_len $PRED_LEN --nonautoregressive \
  --enc_in $ENC_IN_MS --dec_in $DEC_IN_MS --c_out $C_OUT_MS \
  --label_len $LABEL_LEN --pred_len $PRED_LEN \
  --d_model 128 --d_ff 512 --n_heads 8 --e_layers 2 --d_layers 1 \
  --factor 3 --dropout $DROPOUT --activation gelu \
  --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR \
  --train_epochs $TRAIN_EPOCHS --patience $PATIENCE --gpu 0 --use_norm \
  --use_tail_aware_loss --tail_alpha 2.0 --tail_beta 0.003 --tail_mode two_sided \
  --alarm_threshold_high 0.98 --alarm_threshold_low 0.85 \
  --des TCNTransformer-MS
