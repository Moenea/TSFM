# TEP IDV13 Gate-T2 — few-shot workflow

Few-shot counterpart of `../../full_shot/TEP_IDV13`. Same model and evaluation
(sl96 single-token Timer-XL-S, raw + DIFF experts, Gate-T2, IDV13 / XMEAS07,
horizon 15 = 45 min). The ONLY difference: the adaptation training set is a
seeded, random, NESTED subset of the Run1-7 windows. Validation (Run8) and test
(Run9-10) are always full.

This isolates the foundation-model selling point: *how little target-domain fault
data do we need to adapt a pretrained TSFM and still get useful prognosis?*

## Few-shot mechanism

`--data MultivariateDatasetYAMLSplitFewShot --subset_rand_ratio R` keeps a random
`R` fraction of the training windows. Properties (see the class docstring in
`data_provider/data_loader.py`):

- **Random**, not prefix — fault-region windows are represented even at R=0.01
  (~70-77% of windows touch the fault), unlike the base class which would keep
  only the earliest, pre-fault windows of Run1.
- **Nested** — `0.05 ⊂ 0.10 ⊂ ...` via one fixed permutation, so "amount of data"
  is the only variable along the curve.
- **Expert-consistent** — raw and DIFF adapt on the SAME windows.

Scaler / control limits are still fit on full train files (abundant normal-
operation data); the few-shot constraint is on fault-adaptation windows.

## Run

One fraction:

```bash
cd /home/aicode/sherwin/TSFM
SUBSET_RATIO=0.05 bash scripts/adaptation/few_shot/TEP_IDV13/run_all.sh
```

Full sample-count curve (default ratios 0.01 0.02 0.05 0.1 0.25 0.5 1.0):

```bash
bash scripts/adaptation/few_shot/TEP_IDV13/run_curve.sh
# override: RATIOS="0.02 0.1 1.0" GPU_PHYSICAL=1 bash .../run_curve.sh
```

Pass ratios as canonical Python floats (e.g. `0.1`, not `0.10`) so the shell tag
and the aggregator's tag agree.

## Outputs

- per-ratio checkpoints `checkpoints/forecast_..._few_r<tag>_...`
- per-ratio base / gate predictions under `results/...few_r<tag>...`
- per-ratio report `results/TEP_IDV13_XMEAS07_FewShot_Summary/metrics_r<tag>.json`
- aggregated curve `.../FewShot_Summary/curve.{csv,json}` and `curve.png`
