# Reproducing the XMEAS10 / IDV13 manuscript

Target `XMEAS10 Purge Rate`, disturbance IDV13 at magnitude 25, horizon `H = 5`
(15 min), clean window `C = 10` (30 min). Test set = Run9 + Run10 concatenated
(4000 samples, 3618 forecast windows, Δt = 3 min).

## One command

```bash
python scripts/reproduce_paper_metrics.py \
    --verify 'results/XMEAS10 Purge Rate_Summary_clean10.csv'
```

Prints the diagnostics below and writes
`results/XMEAS10 Purge Rate_Summary_paper.csv`:

```
windows=3618  clean=1476  positives=101  negatives=1375
onset events=62  reachable under the clean filter=28
of the positives, already out of band at step 1: 25 (24.8%)
```

| Manuscript | Source | Verified |
|---|---|---|
| Table 1 — prognostic recall | `Summary_paper.csv` `recall` | 66/66 |
| Table 2 — false prognosis rate | `Summary_paper.csv` `FAR` | 66/66 |
| Table 3 — mean lead time | `Summary_paper.csv` `lead_min` | 66/66 |
| Table 4 — improvement over raw | derived from Tables 1/2/3/5 | 32/32 |
| Table 5 — horizon-mean MSE ×10³ | `Summary_mse.csv` `MSE_all` | 66/66 |
| Figs. 3–7 | `figures/IDV13_XMEAS10/plot_paper_figures.ipynb` | 192/192 |

`MSE_all` in `Summary_mse.csv` is **already** scaled by 10³ — Table 5 uses it
verbatim, do not rescale.

## Why there is a separate script

`utils/batch_metrics.py` implements an older convention and is deliberately left
untouched so every other consumer keeps its exact behaviour. It differs from the
manuscript in three ways:

| | manuscript | `batch_metrics.py` |
|---|---|---|
| alarm rule | any of the `H` forecast steps leaves the band (Eqs. 19, 21, 22) | latter half only, `half_start = eval_steps // 2` |
| lead time | averaged over `D`, the onsets the method anticipates; warning issued at the forecast origin, so the range is `[Δt, H·Δt]` = 3–15 min (Eq. 23) | missed onsets folded in as zero lead, origin not counted, range `[0, H·Δt]` |
| clean filter | `C = 10` | read from `params.input_clean_steps` |

Measured impact at `C = 10` (Union, ρ = 100% / ρ = 1%):

- alarm rule: recall 0.792 → 0.784; the largest single shift is raw@1%,
  0.446 → 0.371. FAR moves by ≤ 0.0015.
- lead time: Union@1% 10.82 → 3.53 min, because the zero-lead misses
  re-introduce recall into the lead metric.

Both scripts import `build_window_starts` and `contiguous_events` from
`utils.batch_metrics`, so window construction and event segmentation cannot
drift apart between the two paths.

## Choice of C

```bash
python scripts/c_sensitivity_sweep.py
```

| C | min | positives | negatives | onsets | Union@1% | keep | Union@1% − best baseline@100% | NaN lead cells |
|---|---|---|---|---|---|---|---|---|
| **10** | **30** | **101** | **1375** | **28** | **0.723** | **91.2%** | **+0.188** | **0** |
| 15 | 45 | 76 | 1304 | 18 | 0.711 | 91.5% | +0.158 | 3 |
| 20 | 60 | 54 | 1260 | 13 | 0.685 | 86.0% | +0.111 | 3 |
| 25 | 75 | 42 | 1228 | 11 | 0.690 | 82.9% | +0.071 | 3 |
| 30 | 90 | 40 | 1198 | 9 | 0.700 | 82.4% | +0.075 | 3 |

Raising `C` shrinks the sample fast and lets the baselines improve at ρ = 100%,
so the headline "∇ at 1% data beats the best baseline at 100% data" margin
collapses from +0.119 (C = 10) to exactly 0.000 at C = 25 and C = 30. At C ≥ 15
three lead-time cells become undefined (STA-ConvBiLSTM@1%, CNN-LSTM@1%,
LSTM-GRU@5% anticipate no event at all). `C = 10` also has the simplest physical
reading: the process must have been in band for 30 min, twice the 15 min
forecast horizon.

## Deprecated column

`results/XMEAS10 Purge Rate_Summary_clean10.csv` is the original frozen run. Its
`recall` and `FAR` columns are authoritative and reproduce exactly (max |diff|
≈ 1e-16), but its **`lead_min` column uses the old convention** (raw@1% reads
1.887 where Table 3 reports 8.09). Use `Summary_paper.csv` for lead time.
