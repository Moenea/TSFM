#!/usr/bin/env bash
# Run all three gate-ensemble tiers in sequence (tier1 -> tier2 -> tier3).
# Each tier writes its own results/ensemble_${OUTPUT_NAME}_test_0/ directory.
#
# Environment overrides (forwarded to each tier):
#   SOURCE_A      yaml entry name for source A (default Timer-XL-MS-DIFF)
#   SOURCE_B      yaml entry name for source B (default Timer-XL-MS)
#   OUT_PREFIX    prefix for output_name; tiers append -T{1,2,3}-TXL-MS  (default Gate)
# Tier2/Tier3 only:
#   EPOCHS HIDDEN LR WD SEED DEVICE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_PREFIX=${OUT_PREFIX:-Gate}

echo "=========================================="
echo "Gate-Ensemble fusion: tier1 (closed-form alpha)"
echo "=========================================="
OUTPUT_NAME="${OUT_PREFIX}-T1-TXL-MS" \
  bash "$SCRIPT_DIR/fuse_tier1.sh"

echo "=========================================="
echo "Gate-Ensemble fusion: tier2 (context-only MLP gate)"
echo "=========================================="
OUTPUT_NAME="${OUT_PREFIX}-T2-TXL-MS" \
  bash "$SCRIPT_DIR/fuse_tier2.sh"

echo "=========================================="
echo "Gate-Ensemble fusion: tier3 (stacking MLP)"
echo "=========================================="
OUTPUT_NAME="${OUT_PREFIX}-T3-TXL-MS" \
  bash "$SCRIPT_DIR/fuse_tier3.sh"

echo "=========================================="
echo "All three tiers done. Run batch_metrics_zjsh.sh next to evaluate."
echo "=========================================="
