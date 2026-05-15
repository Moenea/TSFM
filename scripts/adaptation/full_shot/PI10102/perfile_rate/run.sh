#!/usr/bin/env bash
# Per-file Δ-space rate-quantile alarm pipeline driver (bypass tool).
#
# Two stages, fully independent of utils/batch_metrics.py and existing limits:
#   1) build_limits.py            -> per-file τ = quantile(|Δx|, q) limits CSV
#   2) batch_metrics_perfile_rate -> Δ-space per-window alarm metrics + plots
#
# All outputs (metrics, summary, plots, limits copy) land under $OUT_DIR
# (default figures/PI10102/diff). No existing CSV / code / model output is
# touched by this script.
#
# Env overrides:
#   PYTHON_BIN   python interpreter (default tsfm conda env)
#   CONFIG       batch_metrics yaml (default setting/batch_metrics_zjsh_pi10102ts.yaml)
#   QUANTILE     τ quantile (default 0.95)
#   LIMITS_CSV   per-file limits output path
#                (default setting/limits_zjsh_pi10102ts_perfile_rate.csv)
#   OUT_DIR      results root (default figures/PI10102/diff)
#   SKIP_BUILD   if =1, skip build_limits step (reuse existing LIMITS_CSV)

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/aicode/miniconda3/envs/tsfm/bin/python}
CONFIG=${CONFIG:-setting/batch_metrics_zjsh_pi10102ts.yaml}
QUANTILE=${QUANTILE:-0.95}
LIMITS_CSV=${LIMITS_CSV:-setting/limits_zjsh_pi10102ts_perfile_rate.csv}
OUT_DIR=${OUT_DIR:-figures/PI10102/diff}
SKIP_BUILD=${SKIP_BUILD:-0}

cd /home/aicode/sherwin/TSFM

SCRIPT_DIR="scripts/adaptation/full_shot/PI10102/perfile_rate"

echo "=========================================================="
echo "Per-file Δ-space alarm pipeline"
echo "  PYTHON_BIN = $PYTHON_BIN"
echo "  CONFIG     = $CONFIG"
echo "  QUANTILE   = $QUANTILE"
echo "  LIMITS_CSV = $LIMITS_CSV"
echo "  OUT_DIR    = $OUT_DIR"
echo "  SKIP_BUILD = $SKIP_BUILD"
echo "=========================================================="

if [ "$SKIP_BUILD" != "1" ]; then
    echo
    echo "[1/2] Building per-file limits ..."
    "$PYTHON_BIN" -u "$SCRIPT_DIR/build_limits.py" \
        --config "$CONFIG" \
        --quantile "$QUANTILE" \
        --output "$LIMITS_CSV"
else
    echo
    echo "[1/2] SKIP_BUILD=1 -> reusing existing $LIMITS_CSV"
    if [ ! -f "$LIMITS_CSV" ]; then
        echo "ERROR: $LIMITS_CSV does not exist, cannot skip build step." >&2
        exit 1
    fi
fi

echo
echo "[2/2] Computing Δ-space per-file alarm metrics ..."
"$PYTHON_BIN" -u "$SCRIPT_DIR/batch_metrics_perfile_rate.py" \
    --config "$CONFIG" \
    --limits_csv "$LIMITS_CSV" \
    --out_dir "$OUT_DIR"

echo
echo "Done. Results under $OUT_DIR"
