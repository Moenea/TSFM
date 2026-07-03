#!/usr/bin/env bash
set -euo pipefail
# Time-MoE + Sundial zero-shot on univariate XMEAS10 (predict XMEAS10 from its own
# history). These TSFMs are univariate-only, so they do NOT use the other 4 sensors.
# Zero-shot => ratio-independent => run ONCE (reused for every few-shot ratio's gate).
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/aicode/sherwin/TSFM; cd "$ROOT"
FM_PY=${FM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TGT="XMEAS10 Purge Rate"
FE=scripts/adaptation/foundation_experts
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Models are already cached locally (Time-MoE in HF cache, Sundial in checkpoints/sundial_local).
# Force offline so from_pretrained never pings the (flaky) HF hub -> no MaxRetryError.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for split_tag in "setting/TEP_IDV13_XMEAS10.yaml:test" "setting/TEP_IDV13_XMEAS10_val.yaml:val"; do
  sf=${split_tag%%:*}; tag=${split_tag##*:}
  echo "###### Time-MoE zero-shot XMEAS10 ($tag) ######"
  "$FM_PY" "$FE/time_moe/adapter.py" --mode predict --zero-shot \
    --split-file "$sf" --data-root "$DATA_ROOT" --target "$TGT" --horizon 15 \
    --out-dir "results/fm_time_moe_xmeas10_zeroshot_${tag}" --device cuda:0
  echo "###### Sundial zero-shot XMEAS10 ($tag) ######"
  "$FM_PY" "$FE/sundial/adapter.py" --mode predict --zero-shot --num-samples 20 --batch-size 32 \
    --split-file "$sf" --data-root "$DATA_ROOT" --target "$TGT" --horizon 15 \
    --out-dir "results/fm_sundial_xmeas10_zeroshot_${tag}" --device cuda:0
done
echo "DONE FM zero-shot XMEAS10 -> results/fm_{time_moe,sundial}_xmeas10_zeroshot_{val,test}"
