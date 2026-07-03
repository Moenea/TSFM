#!/usr/bin/env bash
# DL baselines (CNNLSTM/DiPCALSTM/LSTMGRU/STAConvBiLSTM/TCNTransformer) on TEP IDV13,
# MS-diff mode, few-shot curve. IDENTICAL splits/windows to the TSFM experts+gate:
#   - MultivariateDatasetYAMLSplitFewShot (seed 2021 subset) on the SAME Run1-7/8/9-10,
#   - seq_len=96, output_token_len=96 -> 1810 windows/file (== TSFM), evaluated at horizon 15.
# MS (all 53 sensors, XMEAS07 last) + first-difference (csv_ms_diff) + cumsum restore to raw.
# (raw baselines collapse to the mean w/o instance-norm; diff is the fair, non-broken setup.)
set -uo pipefail
PY=/home/aicode/miniconda3/envs/tsfm/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ROOT=/home/aicode/sherwin/TSFM; cd "$ROOT"
DATA=/home/aicode/sherwin/dataset/TEP/
SPLIT=setting/TEP_IDV13_MS_diff.yaml
RAWSPLIT=setting/TEP_IDV13_MS.yaml
TGT="XMEAS07 Reactor Pressure"
RATIOS=${RATIOS:-"0.01 0.02 0.05 0.1 0.25 0.5 1.0"}
MODELS=${MODELS:-"CNNLSTM DiPCALSTM LSTMGRU STAConvBiLSTM TCNTransformer"}
LOG=$ROOT/results/TEP_IDV13_baselines_MSdiff.log
: > "$LOG"
declare -A DM=(  [CNNLSTM]=230 [DiPCALSTM]=472 [LSTMGRU]=316 [STAConvBiLSTM]=268 [TCNTransformer]=128 )
declare -A DFF=( [CNNLSTM]=1408 [DiPCALSTM]=2048 [LSTMGRU]=2048 [STAConvBiLSTM]=2048 [TCNTransformer]=512 )
declare -A EL=(  [CNNLSTM]=2 [DiPCALSTM]=2 [LSTMGRU]=1 [STAConvBiLSTM]=1 [TCNTransformer]=2 )
for m in $MODELS; do
  for r in $RATIOS; do
    tag=$(printf 'r%s' "$r" | tr '.' 'p')
    echo "======== $m $tag $(date +%H:%M:%S) ========" | tee -a "$LOG"
    "$PY" -u run.py --task_name long_term_forecast --is_training 1 \
      --root_path "$DATA" --split_file "$SPLIT" \
      --data MultivariateDatasetYAMLSplitFewShot --subset_rand_ratio "$r" \
      --model_id TEP_IDV13_MSdiff_${m}_${tag} --model "$m" \
      --features MS --covariate --last_token --target "$TGT" \
      --restore_diff_to_raw --raw_split_file "$RAWSPLIT" --restore_target "$TGT" \
      --seq_len 96 --input_token_len 96 --output_token_len 96 \
      --test_seq_len 96 --test_pred_len 96 --nonautoregressive \
      --enc_in 53 --dec_in 53 --c_out 1 --label_len 48 --pred_len 96 \
      --d_model ${DM[$m]} --d_ff ${DFF[$m]} --e_layers ${EL[$m]} --d_layers 1 --n_heads 8 --dropout 0.2 \
      --batch_size 128 --num_workers 4 --learning_rate 1e-3 \
      --train_epochs 30 --patience 10 --gpu 0 --cosine --tmax 30 \
      --des ${m}-MSdiff >> "$LOG" 2>&1 \
      && echo "OK   $m $tag" | tee -a "$LOG" \
      || echo "FAIL $m $tag" | tee -a "$LOG"
  done
done
echo "ALL_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
