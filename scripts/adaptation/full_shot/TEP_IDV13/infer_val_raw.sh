#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
inference_args "$RAW_VAL_MODEL_ID" "$RAW_VAL_SPLIT" "$RAW_SETTING" "$RAW_VAL_SETTING"
