#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
# Optional: training already produces the Run9-10 test predictions. This
# regenerates them from the saved checkpoint without retraining.
inference_args "$RAW_MODEL_ID" "$RAW_SPLIT" "$RAW_SETTING" "$RAW_SETTING"
