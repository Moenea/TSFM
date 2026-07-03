#!/usr/bin/env bash
set -euo pipefail
# For one SUBSET_RATIO: fuse the 4 TSFM experts into the gate, then evaluate
# experts + 5 baselines + gate on XMEAS10 (horizon 15) into one metrics_<tag>.json.
# Requires: run_timerxl.sh (this ratio), run_fm_zeroshot.sh (once), run_baselines.sh done.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/_common.sh"   # sets RATIO_TAG, RAW_SETTING, DIFF_SETTING, *_VAL_SETTING, TARGET
FE=$ROOT/scripts/adaptation/foundation_experts
SUMMARY=$ROOT/results/TEP_IDV13_XMEAS10_Summary; mkdir -p "$SUMMARY"
tag=$RATIO_TAG
R=results

MOE_TEST=$R/fm_time_moe_xmeas10_zeroshot_test; MOE_VAL=$R/fm_time_moe_xmeas10_zeroshot_val
SUN_TEST=$R/fm_sundial_xmeas10_zeroshot_test;  SUN_VAL=$R/fm_sundial_xmeas10_zeroshot_val
GATE=$R/ensemble_Gate_XMEAS10_${tag}_test

echo "###### gate fusion (diff, raw, time_moe, sundial) ratio=$SUBSET_RATIO ######"
"$PYTHON_BIN" "$FE/fuse_gate_multi.py" \
  --expert "diff:$R/$DIFF_SETTING" --expert "raw:$R/$RAW_SETTING" \
  --expert "time_moe:$MOE_TEST" --expert "sundial:$SUN_TEST" \
  --val-expert "diff:$R/$DIFF_VAL_SETTING" --val-expert "raw:$R/$RAW_VAL_SETTING" \
  --val-expert "time_moe:$MOE_VAL" --val-expert "sundial:$SUN_VAL" \
  --data-root "$DATA_ROOT" \
  --train-split "$ROOT/setting/TEP_IDV13_XMEAS10.yaml" \
  --val-split "$ROOT/setting/TEP_IDV13_XMEAS10_val.yaml" \
  --test-split "$ROOT/setting/TEP_IDV13_XMEAS10.yaml" \
  --target "$TARGET" --output-dir "$GATE"

# collect baseline result dirs for this ratio (both raw and diff variants)
BE=()
for m in CNNLSTM DiPCALSTM LSTMGRU STAConvBiLSTM TCNTransformer; do
  for mode in raw diff; do
    d=$(ls -d $R/long_term_forecast_TEP_IDV13_XMEAS10_5var_${mode}_${m}_${tag}_*${m}-X10-${mode}_0 2>/dev/null | head -1)
    [ -n "$d" ] && BE+=(--expert "${m}_${mode}:${d}") || echo "WARN missing baseline $m $mode $tag"
  done
done

echo "###### evaluate experts + baselines + gate ######"
"$PYTHON_BIN" "$FE/evaluate_multi.py" \
  --expert "diff:$R/$DIFF_SETTING" --expert "raw:$R/$RAW_SETTING" \
  --expert "time_moe:$MOE_TEST" --expert "sundial:$SUN_TEST" \
  --expert "gate:$GATE" "${BE[@]}" \
  --data-root "$DATA_ROOT" --split "$ROOT/setting/TEP_IDV13_XMEAS10.yaml" \
  --target "$TARGET" --output "$SUMMARY/metrics_${tag}.json"
echo "DONE gate+eval ratio=$SUBSET_RATIO -> $SUMMARY/metrics_${tag}.json"
