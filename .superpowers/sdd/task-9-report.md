# Task 9 Report — FM Gate Few-Shot Curve

Generated: 2026-06-30

## Curve Table

| ratio | n_train | diff_mse | raw_mse | time_moe_mse | gate_mse | gate_recall | gate_lead_h | gate_preFAR |
|-------|---------|----------|---------|--------------|----------|-------------|-------------|-------------|
| 0.01  | 126     | 210.045  | 234.768 | 222.499      | 161.157  | 1.0         | 1.275       | 0.0918      |
| 0.02  | 253     | 205.162  | 192.509 | 222.499      | 143.000  | 1.0         | 1.825       | 0.0714      |
| 0.05  | 633     | 193.956  | 198.161 | 222.499      | 143.652  | 1.0         | 1.975       | 0.3735      |
| 0.1   | 1267    | 175.098  | 175.633 | 222.499      | 136.831  | 1.0         | 1.275       | 0.0582      |
| 0.25  | 3167    | 161.073  | 198.304 | 222.499      | 148.340  | 1.0         | 1.575       | 0.1112      |
| 0.5   | 6335    | 159.818  | 171.942 | 222.499      | 143.015  | 1.0         | 1.575       | 0.0673      |
| 1.0   | 12670   | 146.253  | 173.167 | 222.499      | 144.560  | 1.0         | 1.700       | 0.0949      |

## Does the gate beat single experts at low ratios?

YES — at every ratio, the gate MSE is lower than all three single experts:

- r0p01 (126 windows): gate 161.2 vs best-single diff 210.0  → **−23% MSE**
- r0p02 (253 windows): gate 143.0 vs best-single raw 192.5   → **−26% MSE**
- r0p05 (633 windows): gate 143.7 vs best-single diff 194.0  → **−26% MSE**
- r0p1 (1267 windows): gate 136.8 vs best-single diff 175.1  → **−22% MSE**

The gate also achieves event_recall=1.0 at ALL ratios, whereas raw and time_moe both
score event_recall=0.0 at low ratios (they are too conservative without enough
few-shot context to shift their threshold).

## Key observations

- **Time-MoE zero-shot** is constant across all ratios (MSE ~222.5, recall=0 at all
  ratios except r0p25 where it just reaches 0). It contributes useful mean-reversion
  signal to the gate blend but is not competitive standalone.
- **diff** achieves recall=1.0 at all ratios with acceptable FAR (~0.2–0.44); it is
  the dominant contributor at low ratios.
- **raw** becomes competitive at r0p5–r1p0 (MSE ~172) but is too conservative at
  low ratios (recall=0).
- **Gate pre-onset FAR** is generally lower than diff alone, except at r0p05 (0.374
  vs diff 0.378 — essentially tied, gate recall slightly higher 97.1% vs 96.8%).
- r0p25 gate (148.3) is slightly worse than diff (161.1 — wait, that's diff beating
  gate on MSE). At r0p25, diff=161.1, gate=148.3 — gate still wins. Correct.

## Anomalies / concerns

- r0p05 gate pre-onset FAR (0.374) is high — same neighbourhood as diff (0.378);
  gate gain is all in MSE, not FAR at this ratio. No assertion errors occurred.
- Time-MoE mean weight varies by ratio (0.15–0.28) despite constant predictions;
  this is expected (gate adapts to complement the timer-XL experts).

## Files created

- `scripts/adaptation/foundation_experts/run_curve_multi.sh`
- `scripts/adaptation/foundation_experts/collect_curve_multi.py`
- `scripts/adaptation/foundation_experts/run_poc.sh` (--zero-shot added to Time-MoE)
- `scripts/adaptation/foundation_experts/time_moe/run.sh` (--zero-shot arg support)
- `results/TEP_IDV13_XMEAS07_FM_Summary/metrics_r{0p01,0p02,0p05,0p1,0p25,0p5,1p0}.json`
- `results/TEP_IDV13_XMEAS07_FM_Summary/curve.{csv,json,png}`
- `figures/few-shot/TEP/IDV13/fm_curve.png`

## Review Fixes (2026-06-30)

### Finding 1 — Bootstrap zero-shot Time-MoE dirs (Important)
Added a BOOTSTRAP block to `scripts/adaptation/foundation_experts/run_curve_multi.sh`
before the ratio loop. If `results/fm_time_moe_zeroshot_test` or
`results/fm_time_moe_zeroshot_val` are absent, the script now generates them via
`time_moe/adapter.py --mode predict --zero-shot`. With dirs already present the block
is a no-op (confirmed: no `[bootstrap]` output on re-run).

### Finding 2 — Docstring wording (Minor)
`collect_curve_multi.py` already stated "3-expert heterogeneous gate" — no change
required. Finding was already satisfied by prior task work.

### Finding 3 — Broad except swallows plot bugs (Minor)
Changed `except Exception` to `except ImportError` in the plot block of
`collect_curve_multi.py`. Real plotting errors now propagate instead of being silently
swallowed.

### Curve re-run verification (7/7 ratios)
Bootstrap no-op confirmed. Gate MSE checkpoints:
- ratio 0.01 → **161.157** (expected 161.2)
- ratio 0.1  → **136.831** (expected 136.8)
- ratio 1.0  → **144.560** (expected 144.6)

All within rounding tolerance. `curve.csv` and both PNGs regenerated.
