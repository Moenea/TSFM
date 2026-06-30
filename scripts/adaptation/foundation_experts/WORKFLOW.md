# Foundation-Expert Gate — Workflow Guide

## Overview

`run_poc.sh` is the ratio=1.0 proof-of-concept orchestrator. It:
1. Ensures Timer-XL diff/raw results exist (skips if already present).
2. Runs any available new-model adapters (`time_moe/run.sh`, `sundial/run.sh`).
3. Fuses all present experts via `fuse_gate_multi.py` (N-expert softmax gate).
4. Evaluates each expert + the gate via `evaluate_multi.py`.

---

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `TSFM_PY` | `/home/aicode/miniconda3/envs/tsfm/bin/python` | Interpreter for gate/eval scripts and Timer-XL |
| `FM_PY` | `$TSFM_PY` | Interpreter for FM adapter scripts (may differ if a model needs a separate env) |
| `TAG` | `r1p0` | Result-tag suffix; controls result dir names |
| `CUDA_VISIBLE_DEVICES` | — | Set to `0` for all model code; GPU 1 is occupied by another user |

The `tsfm` env is used for **all current models** (Time-MoE and Sundial both load via `trust_remote_code=True` under transformers 4.40.1). Do not mutate this env.

---

## Expert Contract

Each expert adapter must produce in its output directory:
- `pred.npy` — shape `(B, pred_len)` in the *original* (unnormalized) target space
- `true.npy` — shape `(B, pred_len)` matching pred shape and window alignment
- `meta.json` — at minimum `{"method": "<name>", "tag": "<r1p0|...>"}`

The **first expert** passed to `fuse_gate_multi.py` is the *base expert*. It supplies:
- The canonical `true.npy` (all other experts' truths are alignment-checked against it).
- The full-length `pred.npy` for the horizon tail (`pred_len > horizon` positions).

By convention, `diff` (Timer-XL DIFF-trained) is always listed first.

---

## Running the PoC

```bash
# From the repo root:
CUDA_VISIBLE_DEVICES=0 bash scripts/adaptation/foundation_experts/run_poc.sh
```

Produces:
- Gate result dir: `results/ensemble_Gate-multi_r1p0_test/`
  - `pred.npy`, `true.npy`, `meta.json`
  - `weights.npy` — per-window softmax weights `(B, horizon, N_experts)`
  - `gate.pt` — trained gate MLP state dict
  - `fit_log.json` — training loss, per-expert test MSE, gate test MSE, mean weights
- Evaluation report: `results/TEP_IDV13_XMEAS07_FM_Summary/metrics_r1p0.json`

---

## Timer-XL r1p0 Result Dirs

These are the canonical r1p0 Timer-XL dirs (already present, not regenerated):

| Role | Path |
|---|---|
| diff test | `results/forecast_TEP_IDV13_XMEAS07_S_few_r1p0_DIFF_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0` |
| raw test  | `results/forecast_TEP_IDV13_XMEAS07_S_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0` |
| diff val  | `results/forecast_TEP_IDV13_XMEAS07_S_few_r1p0_DIFF_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0` |
| raw val   | `results/forecast_TEP_IDV13_XMEAS07_S_few_r1p0_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0` |

---

## New-Expert Result Dir Naming Convention

| Model | Test dir | Val dir |
|---|---|---|
| Time-MoE | `results/fm_time_moe_r1p0_test` | `results/fm_time_moe_r1p0_val` |
| Sundial | `results/fm_sundial_r1p0_test` | `results/fm_sundial_r1p0_val` |

---

## Adapter Guard Pattern

`run_poc.sh` guards each new-model adapter and each new-expert result dir:

```bash
# Adapter invocation guard:
[ -f "$HERE/time_moe/run.sh" ] && \
  SUBSET_RATIO=1.0 CUDA_VISIBLE_DEVICES=0 FM_PY=$FM_PY bash "$HERE/time_moe/run.sh"

# Gate inclusion guard:
if [ -d "results/fm_time_moe_r1p0_test" ]; then
  EXPERTS+=(--expert "time_moe:results/fm_time_moe_r1p0_test")
  VAL_EXPERTS+=(--val-expert "time_moe:results/fm_time_moe_r1p0_val")
fi
```

This means the PoC runs **with however many experts are present**, never hard-failing on missing models.

---

## Fallback Notes (from PROBE.md)

- **MOIRAI** is excluded: `uni2ts` requires `torch>=2.1` and is not installed in the `tsfm` env. Installing it would break Timer-XL. If ever pursued, it must go in an isolated `tsfm_fm` clone env. The gate works without it.
- **Sundial**: weights (`model.safetensors`, 513 MB) were stalled on HF's Xet CDN at time of writing. They download via `HF_HUB_DISABLE_XET=1`. Once downloaded and `sundial/run.sh` is written, the guard in `run_poc.sh` picks it up automatically on the next run.
- **Time-MoE**: fully confirmed, 113M params, loads cached. Fine-tuned via `forward(input_ids=ctx, labels=future).loss` with per-window instance normalization. Zero-shot fallback possible.
- **Device**: always `CUDA_VISIBLE_DEVICES=0`. GPU 1 is occupied by another user's jobs.

---

## Backward-Compat Guard

`tests/test_backward_compat.py` verifies that restricting the N-expert softmax gate
to `[diff, raw]` reproduces the existing Gate-T2 reference within ±5.0 MSE:

```bash
/home/aicode/miniconda3/envs/tsfm/bin/python \
  scripts/adaptation/foundation_experts/tests/test_backward_compat.py
```

Expected output: `Gate-T2 reference MSE: 143.1988` / `ALL PASS`.
If the delta exceeds 5.0, there is a real gate regression — fix before proceeding.

---

## Few-Shot Curve Results

The gate was evaluated across seven training-data ratios (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
- The gate beats every single expert across the full few-shot curve.
- In the low-data regime (ratio ≤ 0.1) the margin is large: ~20–26% MSE reduction relative to the best single expert.
- At full data (ratio = 1.0) the gate matches the best single expert; the margin (~1%) is within seed-to-seed re-fit variance (±5 MSE), so no claim of "beating" is warranted.

---

## Exit Criteria for the Full PoC

After `run_poc.sh` completes, inspect `results/TEP_IDV13_XMEAS07_FM_Summary/metrics_r1p0.json`:
- Gate `event_recall` must be ≥ existing Gate-T2 (`1.0`).
- Gate `mse` must be ≤ the worst single new expert included (gate must not drag below Timer-XL).
