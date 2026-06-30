# Heterogeneous-TSFM Gate for TEP Few-Shot Fault Prognosis — Design

- **Date:** 2026-06-30
- **Status:** Approved (design); implementation plan pending
- **Author:** (TEP IDV13 prognosis project)
- **Related:** `scripts/adaptation/few_shot/TEP_IDV13/`, `scripts/adaptation/full_shot/TEP_IDV13/`,
  `figures/few-shot/TEP/IDV13/foundation.png`

## 1. Motivation

The paper's title is *"A Gate-Based Ensemble of **Time Series Foundation Models** for Few-Shot
Fault Prognosis in Chemical Processes."* The plural ("Models") is currently unsupported: the gate
fuses a single backbone — **Timer-XL** — in two views (raw + first-difference). To honour the
title and strengthen the contribution, we add three additional, architecturally distinct TSFMs as
real experts in the gate **and** report each as a standalone baseline:

- **Time-MoE** (`Maple728/TimeMoE-50M`) — decoder-only mixture-of-experts, autoregressive point forecast.
- **MOIRAI** (`Salesforce/moirai-1.0-R-small`) — masked-encoder, any-variate, probabilistic.
- **Sundial** (`thuml/sundial-base-128m`) — generative / flow-matching, sample-based forecast.

All three are **few-shot fine-tuned on the same seed-2021 subset windows as Timer-XL**, so the
few-shot selling point is preserved and the comparison is controlled.

## 2. Confirmed decisions

| Decision | Choice | Rationale |
|---|---|---|
| Role of new models | **Both experts & baselines** | Fuse into the gate AND report each standalone |
| Adaptation regime | **Few-shot fine-tune all three** | Every expert improves with data; identical subset windows |
| First target | **Proof-of-concept at one setting (ratio = 1.0), then scale to the curve** | De-risk dependencies and alignment before any sweep |
| Integration approach | **A — prediction-level adapters + N-expert gate** | Isolates each model's API/loss behind a uniform `pred.npy` contract; reuses the existing gate seam; never touches the fragile strict-loading `exp_forecast` loop |
| MOIRAI/Sundial fine-tune blocked | **Fallback to zero-shot for that model only** | Keeps the effort moving; the gate still fuses it; deviation noted explicitly in results |
| Checkpoint sizes | **Smallest first** (TimeMoE-50M, moirai-1.0-R-small, sundial-base-128m) | Fits the 2 GPUs; upgrade by flag if results warrant |

## 3. Goals / non-goals

**Goals**
- Three new TSFMs fine-tuned on the identical few-shot subset windows used by Timer-XL.
- Each produces window-aligned `pred.npy` / `true.npy` on val (Run8) and test (Run9-10).
- A generalized **N-expert softmax gate** fuses 5 experts (Timer-XL-raw, Timer-XL-diff,
  Time-MoE, MOIRAI, Sundial) and reduces exactly to the current Gate-T2 when N = 2.
- Per-expert and gate metrics (MSE/MAE + prognosis metrics) for the baseline table.
- PoC validated end-to-end before any ratio sweep.

**Non-goals (YAGNI)**
- No raw+diff *double view* for the new models — each contributes **one** expert (raw view only).
  Diff views can be added later if motivated.
- No native port into `run.py` / `exp_forecast` (rejected approach B).
- No multi-seed error bars in this spec (separate, deferred task).
- No new datasets / faults beyond IDV13 / XMEAS07.

## 4. Architecture

### 4.1 Component layout

Parallels the existing `full_shot` / `few_shot` folders so the structure is familiar:

```
scripts/adaptation/foundation_experts/
  common/
    expert_io.py        # windowing (wraps MultivariateDatasetYAMLSplitFewShot),
                        # save pred.npy/true.npy in the result-dir format, norm helpers
  time_moe/
    adapter.py          # fit() + predict()
    run.sh
  moirai/
    adapter.py
    run.sh
  sundial/
    adapter.py
    run.sh
  fuse_gate_multi.py    # N-expert softmax gate (generalizes few_shot/.../fuse_gate_t2.py)
  evaluate_multi.py     # per-expert + gate metrics (generalizes evaluate.py) -> baseline table
  run_poc.sh            # ratio=1.0 end-to-end: 3 adapters -> infer val/test -> fuse 5 -> evaluate
  run_curve_multi.sh    # (phase 2) sweep ratios, build the 5-expert few-shot curve
  WORKFLOW.md
```

### 4.2 Uniform expert contract

Every adapter implements the same two phases and writes the same artifacts, so the gate never
needs to know which model produced them:

- **`fit(ratio)`** — load pretrained weights via the model's own API → fine-tune on
  `MultivariateDatasetYAMLSplitFewShot(set_type=train, subset_rand_ratio=ratio)` (the *identical*
  seed-2021 windows Timer-XL uses) → save a fine-tuned checkpoint.
- **`predict(split)`** — iterate val (Run8) / test (Run9-10) windows **in the same order Timer-XL
  uses**, context = 96 → horizon ≥ 15 point forecast → save
  `pred.npy` of shape `(N_windows, pred_len, 1)` and `true.npy` into
  `results/<expert>_few_r<tag>_test_0/`.

`expert_io.py` is the single source of windowing and saving, so alignment is guaranteed **by
construction**. The gate's existing `np.allclose(true_a, true_b)` assert is the safety net.

Point-forecast extraction per model:
- **Time-MoE** — direct autoregressive output (point forecaster).
- **MOIRAI** — predictive **mean** of the output distribution.
- **Sundial** — **mean of 20 generative samples** (matches the existing `Sundial.py` stub).

### 4.3 N-expert gate (the only change to fusion logic)

`fuse_gate_multi.py` generalizes `few_shot/TEP_IDV13/fuse_gate_t2.py`:

- `GateMLP(in_dim=8, hidden=32, horizon=15)` → output reshaped to `(horizon, N)` with a
  **softmax over the N experts** per horizon step (replaces the sigmoid single weight).
- `fused[:, :15] = Σ_i softmax_i ⊙ pred_i[:, :15]`; the tail beyond horizon 15 keeps a chosen
  **base expert** (the diff view, as today).
- CLI: `--expert NAME:RESULT_DIR` repeated N times. The first listed expert is the base for the
  tail and the `true` reference.
- The 8 context features (last value, delta, mean, std, slope, range, dist-to-low, dist-to-high)
  are computed from the input window and are **model-agnostic** — unchanged.
- Fit on Run8, apply to Run9-10 — same protocol as the current gate.
- **Backward-compat check:** N = 2 with (diff, raw) must reproduce the current Gate-T2 numbers
  (full-shot anchor: MSE ≈ 143.20, lead ≈ 2.125 h, preFAR ≈ 0.0786, event_recall = 1.0).

`evaluate_multi.py` scores every expert and the gate (this doubles as the baseline table).

## 5. Per-model adapter details

| Expert | HF id (default, smallest) | Loader | Fine-tune loss | Point output |
|---|---|---|---|---|
| Time-MoE | `Maple728/TimeMoE-50M` | `transformers` + `trust_remote_code` | MSE / Huber on horizon | direct |
| MOIRAI | `Salesforce/moirai-1.0-R-small` | `uni2ts` | NLL (uni2ts finetune module) | predictive mean |
| Sundial | `thuml/sundial-base-128m` | `transformers` + `trust_remote_code` | flow-matching (if exposed) | mean of 20 samples |

- **Common input:** univariate XMEAS07, context = 96, horizon = 15 (gate horizon). Models may emit
  more than 15 steps; the gate uses the first 15, the tail is handled by the base expert.
- **Normalization:** per-window instance norm consistent with each model's expected input; the
  saved `pred.npy` is in the **original reactor-pressure scale** (de-normalized), matching the
  Timer-XL result dirs the gate already consumes.
- **Checkpoints** are cached locally after first download (HF reachable, 200 OK).

### 5.1 Time-MoE (easiest)
Standard supervised fine-tune: feed the 96-step context, predict the horizon, MSE/Huber loss.
Decoder-only, autoregressive; Time-MoE applies its own per-series scaling. No extra dependency
beyond `transformers` + `trust_remote_code`.

### 5.2 MOIRAI (dependency risk)
Requires `pip install uni2ts` (+ `gluonts`), which are **not currently installed**. Fine-tune via
uni2ts's Moirai finetune module (probabilistic NLL); point forecast for the gate = predictive
mean. **Fallback:** if install or fine-tune is blocked, run **zero-shot MOIRAI** and note it.

### 5.3 Sundial (hardest)
Generative / flow-matching. Inference already works in the `Sundial.py` stub
(`model.generate(..., num_samples=20).mean(dim=1)`). Fine-tune depends on whether the HF remote
code exposes a training loss. **Fallback ladder:** full fine-tune → head-only fine-tune →
zero-shot.

## 6. Window alignment & I/O contract

- All adapters use `MultivariateDatasetYAMLSplitFewShot` for train (with subset) and the base
  `MultivariateDatasetYAMLSplit` for val/test (full), via `expert_io.py`.
- Result-dir layout mirrors Timer-XL's: `pred.npy (N, pred_len, 1)`, `true.npy (N, pred_len, 1)`.
- Usable windows per run = `2001 - 96 - 96 + 1 = 1810`; val = 1810 (Run8), test = 3620 (Run9-10).
- The gate asserts ground-truth identity across experts (`np.allclose`) — a hard alignment guard.

## 7. Phasing

### Phase 1 — Proof of concept (ratio = 1.0)
0. **Load-test** all three models in the target env (resolves the `transformers` version question
   before any training). If a newer `transformers` is needed, use an **isolated conda env** so the
   Timer-XL `tsfm` pipeline is untouched.
1. Fine-tune the three adapters on the full train windows.
2. Infer val + test → window-aligned `pred.npy`.
3. Fuse 5 experts with `fuse_gate_multi.py`; sanity-check N = 2 reproduces Gate-T2.
4. `evaluate_multi.py`: **Gate-5 vs Gate-2 vs each expert**.

**Exit criterion:** all five experts produce aligned predictions; the 5-expert gate runs and is
no worse than the 2-expert gate on test MSE and event recall.

### Phase 2 — Scale to the few-shot curve
Wrap adapters with `SUBSET_RATIO`; add `run_curve_multi.sh` over
`RATIOS = 0.01 0.02 0.05 0.1 0.25 0.5 1.0`; refresh `figures/few-shot/TEP/IDV13/`.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `uni2ts` / `gluonts` not installed (MOIRAI) | Install in isolated env; **fallback zero-shot MOIRAI** |
| Sundial fine-tune loss not exposed | Fallback ladder: full → head-only → zero-shot |
| `transformers 4.40.1` too old for remote code | Phase-1 step 0 load-test; **isolated env** if a bump is needed — never break `tsfm` |
| GPU memory | Smallest checkpoints by default; 2 GPUs available |
| Context-length mismatch across models | Fixed at 96 for alignment; longer history is a knob, not a default |
| Single-seed noise in the curve | Out of scope here; flagged for the deferred multi-seed task |

## 9. Success criteria

1. Five experts, each with aligned `pred.npy` on val + test.
2. `fuse_gate_multi.py` with N = 2 reproduces the current Gate-T2 to ~1e-3.
3. Baseline table: MSE/MAE + prognosis metrics (event_recall, mean_lead_time_h,
   pre_onset_false_alarm_rate, window precision/recall) for each expert and the gate.
4. The 5-expert gate is competitive with or better than the 2-expert gate at ratio = 1.0
   (Phase 1), then characterized across the few-shot curve (Phase 2).
5. Any model that fell back to zero-shot is explicitly labelled in the results.

## 10. Open questions

None blocking. Deferred: multi-seed error bars; optional diff views for the new models;
larger checkpoints if small variants underperform.
