#!/usr/bin/env bash
set -euo pipefail
# Leakage-free inference on the fresh mag=25 runs (gate Run11-17, ext test Run18-27).
# Reuses exp-B (mag25 Run1-7) and exp-A (mag100 Run1-7) checkpoints — NO training.
# Zero-shot Time-MoE/Sundial on the fresh runs. Runs on GPU physical 1.
export GPU_PHYSICAL=1
HERE=/home/aicode/sherwin/TSFM/scripts/adaptation/xmeas10
source "$HERE/_common.sh"
FE=$ROOT/scripts/adaptation/foundation_experts; R=results
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BSUF=timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
B_RAW=forecast_TEP_IDV13_XMEAS10_5var_raw_train25_few_r1p0_${BSUF}
B_DIFF=forecast_TEP_IDV13_XMEAS10_5var_diff_train25_few_r1p0_${BSUF}
A_RAW=forecast_TEP_IDV13_XMEAS10_5var_raw_few_r1p0_${BSUF}
A_DIFF=forecast_TEP_IDV13_XMEAS10_5var_diff_few_r1p0_${BSUF}
TGT="XMEAS10 Purge Rate"
GR=$ROOT/setting/TEP_IDV13_XMEAS10_5var_gate25.yaml
GD=$ROOT/setting/TEP_IDV13_XMEAS10_5var_diff_gate25.yaml
ER=$ROOT/setting/TEP_IDV13_XMEAS10_5var_testext25.yaml
ED=$ROOT/setting/TEP_IDV13_XMEAS10_5var_diff_testext25.yaml
GM=$ROOT/setting/TEP_IDV13_XMEAS10_gate25.yaml
EM=$ROOT/setting/TEP_IDV13_XMEAS10_testext25.yaml

echo "########## [1] exp-B experts on GATE set (Run11-17) ##########"
inference_args_ms mag25_raw_gate25  "$GR" "$B_RAW"  mag25_raw_gate25
inference_args_ms mag25_diff_gate25 "$GD" "$B_DIFF" mag25_diff_gate25 \
  --restore_diff_to_raw --raw_split_file "$GR" --restore_target "$TGT"

echo "########## [2] exp-B experts on EXT test (Run18-27) ##########"
inference_args_ms mag25_raw_testext25  "$ER" "$B_RAW"  mag25_raw_testext25
inference_args_ms mag25_diff_testext25 "$ED" "$B_DIFF" mag25_diff_testext25 \
  --restore_diff_to_raw --raw_split_file "$ER" --restore_target "$TGT"

echo "########## [3] exp-A (mag100) experts on EXT test (Run18-27) ##########"
inference_args_ms mag100_raw_testext25  "$ER" "$A_RAW"  mag100_raw_testext25
inference_args_ms mag100_diff_testext25 "$ED" "$A_DIFF" mag100_diff_testext25 \
  --restore_diff_to_raw --raw_split_file "$ER" --restore_target "$TGT"

echo "########## [4] zero-shot Time-MoE + Sundial on GATE + EXT ##########"
for tag_split in "gate25:$GM" "testext25:$EM"; do
  tag=${tag_split%%:*}; sf=${tag_split##*:}
  "$PYTHON_BIN" "$FE/time_moe/adapter.py" --mode predict --zero-shot \
    --split-file "$sf" --data-root "$DATA_ROOT" --target "$TGT" --horizon 15 \
    --out-dir "$R/fm_time_moe_xmeas10_${tag}" --device cuda:0
  "$PYTHON_BIN" "$FE/sundial/adapter.py" --mode predict --zero-shot --num-samples 20 --batch-size 32 \
    --split-file "$sf" --data-root "$DATA_ROOT" --target "$TGT" --horizon 15 \
    --out-dir "$R/fm_sundial_xmeas10_${tag}" --device cuda:0
done
echo "ALL DONE fresh-run inference"
