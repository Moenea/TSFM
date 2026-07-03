#!/usr/bin/env bash
set -euo pipefail
# One full few-shot pipeline for the current SUBSET_RATIO.
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/train_raw.sh"
bash "$HERE/train_diff.sh"
bash "$HERE/infer_val_raw.sh"
bash "$HERE/infer_val_diff.sh"
bash "$HERE/run_gate_t2.sh"
