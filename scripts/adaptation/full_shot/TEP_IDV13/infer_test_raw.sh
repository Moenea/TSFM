#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
inference_args "$RAW_MODEL_ID" "$RAW_SPLIT" "$RAW_SETTING" "$RAW_SETTING"
