#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
timer_args "$RAW_MODEL_ID" "$RAW_SPLIT"
