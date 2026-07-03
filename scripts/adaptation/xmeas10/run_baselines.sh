#!/usr/bin/env bash
# 5 DL baselines on XMEAS10 (Purge flow), MS, 5 variables, BOTH raw and diff, few-shot curve.
# Same 5 variables as the Timer-XL experts -> fair 1-vs-1 multivariate comparison.
# Identical splits/windows to the TSFM experts+gate (seed-2021 subset, sl96, ot96 -> 1810/file).
#   raw  : train on csv_5var           (predict XMEAS10 directly)
#   diff : train on csv_5var_diff + cumsum-restore to raw XMEAS10
# (raw baselines are expected to collapse to the mean without instance-norm; diff should work.)
set -uo pipefail
PY=/home/aicode/miniconda3/envs/tsfm/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ROOT=/home/aicode/sherwin/TSFM; cd "$ROOT"
DATA=/home/aicode/sherwin/dataset/TEP/
RAWSPLIT=setting/TEP_IDV13_XMEAS10_5var.yaml
DIFFSPLIT=setting/TEP_IDV13_XMEAS10_5var_diff.yaml
TGT="XMEAS10 Purge Rate"
RATIOS=${RATIOS:-"0.01 0.02 0.05 0.1 0.25 0.5 1.0"}
MODELS=${MODELS:-"CNNLSTM DiPCALSTM LSTMGRU STAConvBiLSTM TCNTransformer"}
MODES=${MODES:-"raw diff"}
LOG=$ROOT/results/TEP_IDV13_XMEAS10_baselines.log
: > "$LOG"
declare -A DM=(  [CNNLSTM]=230 [DiPCALSTM]=472 [LSTMGRU]=316 [STAConvBiLSTM]=268 [TCNTransformer]=128 )
declare -A DFF=( [CNNLSTM]=1408 [DiPCALSTM]=2048 [LSTMGRU]=2048 [STAConvBiLSTM]=2048 [TCNTransformer]=512 )
declare -A EL=(  [CNNLSTM]=2 [DiPCALSTM]=2 [LSTMGRU]=1 [STAConvBiLSTM]=1 [TCNTransformer]=2 )

run_one() {  # $1=model $2=ratio $3=mode
  local m=$1 r=$2 mode=$3
  local tag; tag=$(printf 'r%s' "$r" | tr '.' 'p')
  local split
  local restore=()
  if [ "$mode" = "diff" ]; then
    split=$DIFFSPLIT
    restore=(--restore_diff_to_raw --raw_split_file "$RAWSPLIT" --restore_target "$TGT")
  else
    split=$RAWSPLIT
  fi
  echo "======== $m $mode $tag $(date +%H:%M:%S) ========" | tee -a "$LOG"
  "$PY" -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path "$DATA" --split_file "$split" \
    --data MultivariateDatasetYAMLSplitFewShot --subset_rand_ratio "$r" \
    --model_id TEP_IDV13_XMEAS10_5var_${mode}_${m}_${tag} --model "$m" \
    --features MS --covariate --last_token --target "$TGT" "${restore[@]}" \
    --seq_len 96 --input_token_len 96 --output_token_len 96 \
    --test_seq_len 96 --test_pred_len 96 --nonautoregressive \
    --enc_in 5 --dec_in 5 --c_out 1 --label_len 48 --pred_len 96 \
    --d_model ${DM[$m]} --d_ff ${DFF[$m]} --e_layers ${EL[$m]} --d_layers 1 --n_heads 8 --dropout 0.2 \
    --batch_size 128 --num_workers 4 --learning_rate 1e-3 \
    --train_epochs 30 --patience 10 --gpu 0 --cosine --tmax 30 \
    --des ${m}-X10-${mode} >> "$LOG" 2>&1 \
    && echo "OK   $m $mode $tag" | tee -a "$LOG" \
    || echo "FAIL $m $mode $tag" | tee -a "$LOG"
}

for m in $MODELS; do
  for mode in $MODES; do
    for r in $RATIOS; do run_one "$m" "$r" "$mode"; done
  done
done
echo "ALL_DONE $(date +%H:%M:%S)  ($(grep -c '^OK ' "$LOG") OK / $(grep -c '^FAIL ' "$LOG") FAIL)" | tee -a "$LOG"
