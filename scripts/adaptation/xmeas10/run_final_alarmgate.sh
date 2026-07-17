#!/usr/bin/env bash
set -euo pipefail
# T7: train the final alarm gate (CV-winning hparams) on Run9-10 and the extended
# test; build exp-A / exp-B gates on the extended test via the UNCHANGED
# fuse_gate_multi.py; batch_metrics (mag=25 limits) on both sets; 3-way compare.
cd /home/aicode/sherwin/TSFM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/aicode/miniconda3/envs/tsfm/bin/python
R=results; FE=scripts/adaptation/foundation_experts
BSUF=timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
LIM=setting/limits_tep_xmeas10_mag25.csv
DATA=/home/aicode/sherwin/dataset/TEP
TGT="XMEAS10 Purge Rate"

# --- CV winner ---
read TAU LF LL < <($PY - <<'PY'
import pandas as pd
d=pd.read_csv("results/TEP_IDV13_XMEAS10_Summary/alarmgate_cv.csv").sort_values("S",ascending=False).iloc[0]
print(d.tau_soft, d.lambda_far, d.lambda_lead)
PY
)
echo "CV winner: tau_soft=$TAU lambda_far=$LF lambda_lead=$LL"

# expert dirs
B_RAWV=$R/forecast_TEP_IDV13_XMEAS10_5var_raw_val_train25_few_r1p0_${BSUF}
B_DIFFV=$R/forecast_TEP_IDV13_XMEAS10_5var_diff_val_train25_few_r1p0_${BSUF}
A_RAWV=$R/forecast_TEP_IDV13_XMEAS10_5var_raw_val_few_r1p0_${BSUF}
A_DIFFV=$R/forecast_TEP_IDV13_XMEAS10_5var_diff_val_few_r1p0_${BSUF}

# --- final alarm gate: train on 7 gate runs (Run11-17), apply to Run9-10 and ext ---
alarm() {  # $1 test-split  $2 test-expert-tag(mag25_*_<tag> / fm_*_<tag>)  $3 out
  local ts=$1 tag=$2 out=$3
  $PY $FE/fuse_gate_alarm.py \
    --expert diff:$R/mag25_diff_${tag} --expert raw:$R/mag25_raw_${tag} \
    --expert time_moe:$R/fm_time_moe_xmeas10_${tag} --expert sundial:$R/fm_sundial_xmeas10_${tag} \
    --val-expert diff:$R/mag25_diff_gate25 --val-expert raw:$R/mag25_raw_gate25 \
    --val-expert time_moe:$R/fm_time_moe_xmeas10_gate25 --val-expert sundial:$R/fm_sundial_xmeas10_gate25 \
    --data-root $DATA --train-split setting/TEP_IDV13_XMEAS10.yaml --limits-csv $LIM \
    --val-split setting/TEP_IDV13_XMEAS10_gate25.yaml --test-split $ts \
    --target "$TGT" --output-dir $out \
    --tau-soft $TAU --lambda-far $LF --lambda-lead $LL --tau-a 0.003 --epochs 1500
}
echo "### final alarm gate on Run9-10 ###"
# Run9-10 test-experts MUST be the exp-B (mag25-trained) experts to match the gate's
# training distribution (gate25 = exp-B experts). mag25_*_test are exp-A (mag100) experts.
B_RAW_TEST=$R/forecast_TEP_IDV13_XMEAS10_5var_raw_train25_few_r1p0_${BSUF}
B_DIFF_TEST=$R/forecast_TEP_IDV13_XMEAS10_5var_diff_train25_few_r1p0_${BSUF}
$PY $FE/fuse_gate_alarm.py \
  --expert diff:$B_DIFF_TEST --expert raw:$B_RAW_TEST \
  --expert time_moe:$R/fm_time_moe_xmeas10_mag25_test --expert sundial:$R/fm_sundial_xmeas10_mag25_test \
  --val-expert diff:$R/mag25_diff_gate25 --val-expert raw:$R/mag25_raw_gate25 \
  --val-expert time_moe:$R/fm_time_moe_xmeas10_gate25 --val-expert sundial:$R/fm_sundial_xmeas10_gate25 \
  --data-root $DATA --train-split setting/TEP_IDV13_XMEAS10.yaml --limits-csv $LIM \
  --val-split setting/TEP_IDV13_XMEAS10_gate25.yaml --test-split setting/TEP_IDV13_XMEAS10_mag25.yaml \
  --target "$TGT" --output-dir $R/ensemble_Gate_alarm_XMEAS10_test \
  --tau-soft $TAU --lambda-far $LF --lambda-lead $LL --tau-a 0.003 --epochs 1500
echo "### final alarm gate on ext ###"
alarm setting/TEP_IDV13_XMEAS10_testext25.yaml testext25 $R/ensemble_Gate_alarm_XMEAS10_testext

# --- exp-B and exp-A MSE gates on the extended test (existing fuse_gate_multi.py) ---
echo "### exp-B gate on ext ###"
$PY $FE/fuse_gate_multi.py \
  --expert diff:$R/mag25_diff_testext25 --expert raw:$R/mag25_raw_testext25 \
  --expert time_moe:$R/fm_time_moe_xmeas10_testext25 --expert sundial:$R/fm_sundial_xmeas10_testext25 \
  --val-expert diff:$B_DIFFV --val-expert raw:$B_RAWV \
  --val-expert time_moe:$R/fm_time_moe_xmeas10_mag25_val --val-expert sundial:$R/fm_sundial_xmeas10_mag25_val \
  --data-root $DATA --train-split setting/TEP_IDV13_XMEAS10.yaml \
  --val-split setting/TEP_IDV13_XMEAS10_5var_val_train25.yaml --test-split setting/TEP_IDV13_XMEAS10_testext25.yaml \
  --target "$TGT" --output-dir $R/ensemble_Gate_expB_XMEAS10_testext
echo "### exp-A gate on ext ###"
$PY $FE/fuse_gate_multi.py \
  --expert diff:$R/mag100_diff_testext25 --expert raw:$R/mag100_raw_testext25 \
  --expert time_moe:$R/fm_time_moe_xmeas10_testext25 --expert sundial:$R/fm_sundial_xmeas10_testext25 \
  --val-expert diff:$A_DIFFV --val-expert raw:$A_RAWV \
  --val-expert time_moe:$R/fm_time_moe_xmeas10_zeroshot_val --val-expert sundial:$R/fm_sundial_xmeas10_zeroshot_val \
  --data-root $DATA --train-split setting/TEP_IDV13_XMEAS10.yaml \
  --val-split setting/TEP_IDV13_XMEAS10_val.yaml --test-split setting/TEP_IDV13_XMEAS10_testext25.yaml \
  --target "$TGT" --output-dir $R/ensemble_Gate_expA_XMEAS10_testext

# --- batch_metrics (mag25 limits) on both sets ---
mkcfg() {  # $1 out.yaml  $2 test-csv-run-list(space)  $3.. "name:dir" entries
  local out=$1; shift; local runs=$1; shift
  { echo "params:"; echo "  target: $TGT"; echo "  limit_csv_path: $(pwd)/$LIM";
    echo "  data_root: $DATA/csv_5var_lowmag"; echo "  results_root: ./results";
    echo "  seq_len: 96"; echo "  pred_len: 96"; echo "  eval_steps: 15";
    echo "  input_clean_steps: 30"; echo "  alarm_quality_rmse_factor: 0.2"; echo "test:";
    for r in $runs; do echo "- Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run${r}.csv"; done
    echo "model_dirs:"; for e in "$@"; do echo "- {name: ${e%%:*}, result_dir: ${e##*:}}"; done
  } > "$out"
}
mkcfg setting/bm_final910.yaml "9 10" \
  "Gate-expA:ensemble_Gate_XMEAS10_mag25_test" "Gate-expB:ensemble_Gate_XMEAS10_train25_test" \
  "Gate-alarm:ensemble_Gate_alarm_XMEAS10_test"
mkcfg setting/bm_finalext.yaml "18 19 20 21 22 23 24 25 26 27" \
  "Gate-expA:ensemble_Gate_expA_XMEAS10_testext" "Gate-expB:ensemble_Gate_expB_XMEAS10_testext" \
  "Gate-alarm:ensemble_Gate_alarm_XMEAS10_testext"
$PY -u utils/batch_metrics.py --config setting/bm_final910.yaml --summary-suffix _final910 --figure-suffix _final910 >/dev/null 2>&1
$PY -u utils/batch_metrics.py --config setting/bm_finalext.yaml --summary-suffix _finalext --figure-suffix _finalext >/dev/null 2>&1

$PY scripts/adaptation/xmeas10/compare_3way.py
echo "FINAL_DONE"
