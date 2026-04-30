#!/usr/bin/env bash
# Inference for all 8 MyTimeXer baselines × {MS, S} on PI10102 transitions.
#
# Reuses the same hyperparameters as the per-model training scripts so that
# the constructed setting string matches the checkpoint folder created by
# training. Results land in:
#   ./checkpoints/<setting>/checkpoint.pth     (loaded)
#   ./results/<setting>/{pred,true,metrics}.npy (written)
#
# Skip a baseline by setting SKIP_MODELS env var, e.g.
#   SKIP_MODELS="TimeXer GTProger" bash infer_baselines_all.sh
# Run only one mode by setting ONLY_MODE=MS or ONLY_MODE=S.

source "$(dirname "$0")/_common.sh"

# Setting fields fixed across all baselines (must match training scripts):
#   data=MultivariateDatasetYAMLSplit, lr 1e-3 -> 0.001, bt=128, wd=0,
#   nh=8 (default or explicit), cosine flag absent -> cosFalse, ii=0.
LR_PRINT=0.001
WD=0
COS=False
ITER=0

# (model d_model d_ff e_layers tail_aware?)
configs=(
  "CNNLSTM       230 1408 2 0"
  "DiPCALSTM     472 2048 2 0"
  "LSTMGRU       316 2048 1 0"
  "STAConvBiLSTM 268 2048 1 0"
  "TCNTransformer 128 512  2 1"
  "TimeXer        248 992  1 0"
  "GTProger       248 992  1 1"
  "GTProgerV13    248 992  1 0"
)

modes=("MS" "S")
[ -n "$ONLY_MODE" ] && modes=("$ONLY_MODE")

for cfg in "${configs[@]}"; do
  read -r MODEL DM DFF EL TAIL_FLAG <<< "$cfg"

  if [[ " $SKIP_MODELS " == *" $MODEL "* ]]; then
    echo "[skip] $MODEL (in SKIP_MODELS)"; continue
  fi

  for MODE in "${modes[@]}"; do
    # TimeXer / GTProger / GTProgerV13 require >=1 exogenous variable
    # (their cross-attention path crashes with empty ex tensor in S mode).
    if [ "$MODE" = "S" ] && [[ " TimeXer GTProger GTProgerV13 " == *" $MODEL "* ]]; then
      echo "[skip] $MODEL S — model requires exogenous variables"
      continue
    fi
    if [ "$MODE" = "MS" ]; then
      ENC_IN=$ENC_IN_MS; DEC_IN=$DEC_IN_MS; C_OUT=$C_OUT_MS
      MODE_FLAGS="--features MS --covariate --last_token"
    else
      ENC_IN=$ENC_IN_S; DEC_IN=$DEC_IN_S; C_OUT=$C_OUT_S
      MODE_FLAGS="--features S"
    fi

    DES="${MODEL}-${MODE}"
    MODEL_ID="ZJSH_PI10102TS_${MODEL}_${MODE}"

    SETTING="long_term_forecast_${MODEL_ID}_${MODEL}_MultivariateDatasetYAMLSplit"
    SETTING+="_sl${SEQ_LEN}_it${PRED_LEN}_ot${PRED_LEN}"
    SETTING+="_lr${LR_PRINT}_bt${BATCH_SIZE}_wd${WD}"
    SETTING+="_el${EL}_dm${DM}_dff${DFF}_nh8"
    SETTING+="_cos${COS}_${DES}_${ITER}"

    CKPT="./checkpoints/${SETTING}/checkpoint.pth"
    if [ ! -f "$CKPT" ]; then
      echo "[skip] $MODEL $MODE — checkpoint missing at $CKPT"
      continue
    fi

    TAIL_FLAGS=""
    if [ "$TAIL_FLAG" = "1" ]; then
      TAIL_FLAGS="--use_tail_aware_loss --tail_alpha 2.0 --tail_beta 0.003 --tail_mode two_sided --alarm_threshold_high 0.98 --alarm_threshold_low 0.85"
    fi

    # TCNTransformer also passed --d_layers 1 --activation gelu during training,
    # but those don't affect the setting string. Pass them again for safety.
    EXTRA_MODEL_FLAGS=""
    if [ "$MODEL" = "TCNTransformer" ]; then
      EXTRA_MODEL_FLAGS="--d_layers 1 --activation gelu"
    fi

    echo "===== Inference: $MODEL $MODE ====="
    echo "       setting: $SETTING"
    python -u run.py \
      --task_name long_term_forecast --is_training 0 \
      --root_path "$ROOT_PATH" --split_file "$SPLIT_FILE" \
      --data MultivariateDatasetYAMLSplit \
      --model_id "$MODEL_ID" --model "$MODEL" \
      $MODE_FLAGS \
      --seq_len $SEQ_LEN --input_token_len $PRED_LEN --output_token_len $PRED_LEN \
      --test_seq_len $SEQ_LEN --test_pred_len $PRED_LEN --nonautoregressive \
      --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
      --label_len $LABEL_LEN --pred_len $PRED_LEN --patch_len $PATCH_LEN \
      --d_model $DM --d_ff $DFF --n_heads 8 --e_layers $EL --factor 3 --dropout $DROPOUT \
      $EXTRA_MODEL_FLAGS \
      --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR \
      --train_epochs $TRAIN_EPOCHS --patience $PATIENCE --gpu 0 --use_norm \
      $TAIL_FLAGS \
      --des "$DES" \
      --test_dir "$SETTING" --test_file_name checkpoint.pth \
      || echo "[FAIL] $MODEL $MODE"
  done
done

echo "All done."
