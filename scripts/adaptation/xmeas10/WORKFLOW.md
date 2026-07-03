# XMEAS10 (Purge flow) few-shot study — commands to run

Predict **XMEAS10 (Purge Rate)** at horizon 15, few-shot on TEP IDV13. Splits are
**identical** to the XMEAS07 study (Run1-7 train / Run8 val / Run9-10 test, seed-2021
subset, seq_len=96, output_token_len=96 → 1810 windows/file). Ratios: `0.01 0.02 0.05 0.1 0.25 0.5 1.0`.

**Per-method input (deliberately fair):**

| method | input | mode |
|---|---|---|
| Timer-XL raw / diff | 5 vars {XMEAS10,15,17,XMV06,XMV11} → XMEAS10 | MS (`enc_in=5`, `--covariate --last_token`) |
| Time-MoE / Sundial | XMEAS10 only → XMEAS10 | univariate (model limitation) |
| CNNLSTM/DiPCALSTM/LSTMGRU/STAConvBiLSTM/TCNTransformer | same 5 vars → XMEAS10 | MS, **both raw & diff** (`enc_in=5`) |
| Gate | fuse experts' XMEAS10 predictions | — |

All runs use `CUDA_VISIBLE_DEVICES=0` (GPU 1 is shared). Interpreter: the `tsfm` env.

---

## Step 0 — data (already prepared; idempotent)
```bash
cd /home/aicode/sherwin/TSFM
/home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/xmeas10/prepare_5var_diff.py
```
Creates `dataset/TEP/csv_5var/` and `csv_5var_diff/` (5 cols, XMEAS10 last). YAMLs already in `setting/TEP_IDV13_XMEAS10*`.

## Step 1 — SMOKE FIRST (verify Timer-XL MS works before the full sweep)
Timer-XL in MS mode is the one unverified path. Run a 2-epoch check at ratio 1.0:
```bash
SUBSET_RATIO=1.0 EPOCHS=2 bash scripts/adaptation/xmeas10/run_timerxl.sh
```
Then confirm the raw test predictions are XMEAS10-scale, right shape, finite:
```bash
/home/aicode/miniconda3/envs/tsfm/bin/python - <<'PY'
import numpy as np, glob
d=glob.glob("results/forecast_TEP_IDV13_XMEAS10_5var_raw_few_r1p0_timer_xl_*test_0")[0]
p=np.load(d+"/pred.npy"); t=np.load(d+"/true.npy")
print("pred", p.shape, "true", t.shape, "pred.mean %.2f"%p.mean(), "finite", bool(np.isfinite(p).all()))
# expect ~3620 test windows; XMEAS10 (Purge Rate) is O(0.2-0.5) scale
PY
```
If shape is `(3620, 96)`/`(3620,96,1)` and values are finite in the Purge-flow range → good. (If it errors on MS, tell me and I'll adjust `c_out`/covariate.)

## Step 2 — Time-MoE + Sundial zero-shot (once; ratio-independent)
```bash
bash scripts/adaptation/xmeas10/run_fm_zeroshot.sh
```

## Step 3 — 5 DL baselines, MS, BOTH raw & diff (long: ~70 runs)
```bash
bash scripts/adaptation/xmeas10/run_baselines.sh
# progress: grep -c '^OK ' results/TEP_IDV13_XMEAS10_baselines.log   # ../70
# (raw baselines are expected to collapse to ~mean; diff should be competitive.)
```

## Step 4 — full curve: Timer-XL + gate + eval per ratio, then aggregate
(Steps 2 and 3 must be done first.)
```bash
bash scripts/adaptation/xmeas10/run_curve.sh          # all 7 ratios
# or one ratio at a time:
#   SUBSET_RATIO=0.1 bash scripts/adaptation/xmeas10/run_timerxl.sh
#   SUBSET_RATIO=0.1 bash scripts/adaptation/xmeas10/run_gate_eval.sh
```
Outputs:
- `results/TEP_IDV13_XMEAS10_Summary/comparison.csv` (MSE + event_recall, all methods × ratios)
- `figures/few-shot/TEP/IDV13_XMEAS10/baselines_vs_gate.png`

## Notes
- Full Timer-XL runs: use the default `EPOCHS` (10) — omit the `EPOCHS=2` from the smoke.
- The gate fuses diff (base) + raw + time_moe + sundial; baselines are evaluated alongside, not fused.
- If you only want to re-aggregate after runs exist: `python scripts/adaptation/xmeas10/collect_compare.py`.
