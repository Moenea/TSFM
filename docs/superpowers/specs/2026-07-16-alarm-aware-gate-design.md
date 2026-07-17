# Alarm-aware combined-score gate for XMEAS10 / IDV13 mag=25

Date: 2026-07-16
Status: design approved (verbal), pending spec review → implementation plan

## 1. Goal & motivation

Make the **gate** the best prognosis method on the mild TEP IDV13 fault (mag=25):
higher clean-window **recall**, lower **false prognosis rate (FAR)**, longer **mean
lead time** — all three simultaneously beating the current best gate (exp A).

Root cause of the current shortfall (established this session): the gate is a
softmax-over-experts MLP trained to minimize **MSE**. On a mild fault, the
MSE-optimal fused forecast is the *smooth/conservative* blend, which averages
away the `diff` expert's excursions and rarely crosses ±3σ. MSE and alarm-recall
objectives diverge. Retraining everything on mag=25 (exp B) made the gate *more*
conservative (mean `diff` weight 0.44→0.28, clean recall 0.40→0.26, event_recall
0.5→0.0), confirming the objective — not the data magnitude — is the problem.

## 2. Constraints (fixed, non-negotiable)

- **±3σ is a hard decision threshold.** No separate/tuned pre-alarm band. The
  prognosis event is "the forecast crosses ±3σ within the horizon"; the true
  event is "the true signal crosses ±3σ." Both thresholds stay at 3σ.
- **Control limits stay from mag=100 pre-onset** (`setting/limits_tep_xmeas10.csv`,
  LCL=0.180194, UCL=0.243756), identical to exp A / exp B → results comparable.
- **Zero-risk / bit-identical rule.** Do NOT edit the validated
  `fuse_gate_multi.py`, `evaluate_multi.py`, `batch_metrics.py`, or any existing
  result. All new behavior lives in NEW files (`fuse_gate_alarm.py`, generation
  and driver scripts, new split YAMLs, new result dirs). exp A / exp B artifacts
  remain untouched.
- **Only mag=25 data** (per user; drop the earlier multi-magnitude idea).

## 3. Success criterion

Selection metric = a **weighted combined score** of the three clean-window
prognosis metrics (recall↑, (1−FAR)↑, lead↑). Acceptance: on the test set the new
gate must have clean-window **recall ≥, FAR ≤, lead ≥** exp A (dominate on all
three) and the highest combined score among {exp A, exp B, new gate}. If it
cannot dominate all three, fall back (§9).

Combined score (for model selection; weights fixed up front, documented):
`S = w_r·recall_clean + w_f·(1 − FAR_clean/0.05) + w_l·(lead_clean / H)`
with a default `w_r=w_f=w_l=1` and FAR normalized by the 5% industrial budget,
lead normalized by horizon H=15. (Weights are a knob, not a claim; reported.)

## 4. Architecture

Keep the existing gate structure so the downstream metric/plot pipeline is
unchanged (fused output is still one plain forecast curve `pred.npy` of shape
`(N, pred_len, 1)`):

- `GateMLPMulti`: context features (8-dim) → 2×GELU hidden (32) → `H×N` logits →
  **softmax over N=4 experts per horizon step**, with a **temperature `τ_soft`**
  on the logits (`softmax(logits/τ_soft)`). Lower `τ_soft` ⇒ sharper routing ⇒
  near-crossing the gate can approach hard selection of `diff`, so its spike
  survives the blend.
- Context features unchanged (they already include `high−last` = distance to
  upper limit and `slope` — the "crossing imminent" signals the MSE gate ignored).
- Experts N=4: `diff`, `raw`, `time_moe`, `sundial`.

## 5. Loss (the core change) — alarm-aware combined-score surrogate

Per gate-training window `i`, horizon steps `k=1..H` (H=15):

- Fused forecast: `fused[i,k] = Σ_e w[i,k,e] · pred_e[i,k]` (w = tempered softmax).
- **Soft alarm** per step (differentiable ±3σ crossing):
  `a[i,k] = σ((fused[i,k]−UCL)/τ_a) + σ((LCL−fused[i,k])/τ_a)`, clamped to [0,1];
  `τ_a` = alarm sharpness (default ≈ 5% of alarm band).
- **Window soft-alarm** over the latter half (matches metric `half_start=H//2`):
  `A[i] = 1 − Π_{k≥H//2}(1 − a[i,k])` (soft-OR).
- Ground truth per window: `y[i]` = true crosses ±3σ in latter half
  (`true_alarm_last5`); `clean[i]` = input context (30 steps back) alarm-free.
- **Lead weighting** `ℓ[i]` for true-alarm windows: increasing in how far ahead
  the window's forecast origin sits before the true crossing time (earlier ⇒
  larger), normalized to [0,1]. Encourages earlier firing.

Total loss (minimize):
```
L = BCE_alarm                                   # recall + FAR core
  + λ_far  · mean( A[i]        | clean & ¬y )   # extra FAR penalty on clean negatives
  − λ_lead · mean( ℓ[i] · A[i] | y )            # reward early firing on positives
  + λ_mse  · MSE(fused, true)                   # small regularizer, keep forecast sane
```
where `BCE_alarm = mean( −[ y·log A + (1−y)·log(1−A) ] )` over eligible windows
(true-alarm windows + clean negatives). `λ_mse` small and fixed; `λ_far`,
`λ_lead`, `τ_soft`, `τ_a` are swept in selection (§7).

Why it works: the gate has the features to detect "crossing imminent" and, under
this loss (not MSE), is rewarded for routing to `diff` and letting `fused` cross
3σ *then* — and penalized for crossing on clean negatives. Temperature lets the
routing be sharp enough that the excursion survives the convex average
(needs `w_diff` high enough: e.g. to lift 0.235→>0.244 with diff at 0.25 needs
`w_diff>0.6`, reachable with low `τ_soft`).

## 6. Data (disjoint split — no leakage by construction, no OOF)

mag=25 IDV13 has 100 runs in `TEP_h5/TEP_mode1.h5`; Run1–10 already extracted.
Generate fresh runs via `tep_loader.load_fault(mode=1, idv=13, mag=25, run=N)`,
extract the 5 vars `[XMEAS15, XMEAS17, XMV06, XMV11, XMEAS10 Purge Rate]` (target
last) → `csv_5var_lowmag/`, plus first-difference (row0=0) → `csv_5var_lowmag_diff/`
(same convention as `prepare_5var_diff.py`).

| role | runs | size | notes |
|---|---|---|---|
| fine-tune experts (diff/raw) | Run1–7 | 7 runs ≈ 12.6k windows | already trained — reuse exp-B mag=25 checkpoints |
| **gate training** (fresh, disjoint) | Run11–17 | 7 runs ≈ 12.6k windows, ~830 clean positives, 7 events | experts never saw these ⇒ leakage-free; matched to fine-tune size |
| test (comparable) | Run9–10 | 2 runs | preserves comparability with exp A/B |
| test (extended, reliable stats) | Run18–27 | 10 runs, 10 events | event_recall on 2 runs is coarse (0/0.5/1); exp A & exp B gates re-evaluated here too for a fair 3-way comparison |

New runs to generate: Run11–17 (7 gate) + Run18–27 (10 extended test) = 17 fresh
runs, each producing a 5-var raw CSV and a 5-var first-difference CSV.

Leakage argument: experts are fine-tuned ONLY on Run1–7. Gate trains on Run11–17
predictions (generalization-quality, never-trained). Zero-shot Time-MoE/Sundial
never train at all. Test Run9–10 / Run18–27 disjoint from both.

## 7. Model selection (no test-set peeking)

Hyperparameters `{λ_far, λ_lead, τ_soft, τ_a}` (λ_mse fixed) selected by
**grouped cross-validation over the 7 gate runs**: leave-one-gate-run-out, train
gate on 6, score combined-S on the held-out run; average over folds. Pick the
config with best mean CV combined-S that also dominates exp A on all three in CV.
Then retrain the gate on all 7 gate runs with that config → apply to test.

## 8. Evaluation protocol (unchanged machinery)

- `evaluate_multi.py` → all-window metrics (MSE, event_recall, window_recall/prec,
  preFAR, lead_h).
- `batch_metrics.py` with a new config → clean-window metrics
  (`ratio_pred_in_true_alarm_patches_clean` = recall, `..._no_true..._clean` = FAR,
  `mean_lead_time_patch_clean`), same mag=100 limits, `eval_steps=15`,
  `input_clean_steps=30`.
- Report exp A / exp B / new gate + the four experts on the SAME test sets
  (Run9–10 and extended Run18–27), with the combined score.

## 9. Execution phases (de-risked)

- **Phase 0 (fast sanity, ~10 min):** implement `fuse_gate_alarm.py`; train the new
  loss on the EXISTING exp-B Run8 mag=25 data; confirm it beats exp A on the
  clean-window metrics before spending compute on data generation. If not, tune
  the loss first.
- **Phase 1 (full):** generate Run11–17 + Run18–27; leakage-free expert +
  zero-shot inference on them (reusing exp-B Run1–7 checkpoints — NO expert
  retraining); grouped-CV select hyperparameters on the 7 gate runs; retrain gate;
  evaluate all methods on Run9–10 and Run18–27; produce the 3-way comparison.

## 10. Fallbacks

If the alarm-aware softmax gate cannot dominate all three (§3):
1. **Cost-sensitive MSE + temperature** (approach 2): up-weight near-crossing /
   clean-pre-crossing windows in MSE, add `τ_soft`. Lower-risk, smaller push.
2. **Two-headed gate** (approach 3): average head for accuracy + gated soft-vote
   alarm head so a confident `diff` excursion is not averaged away. Changes the
   fused-output semantics; most invasive; only if 1 underdelivers.

## 11. Deliverables

- New scripts: `scripts/adaptation/foundation_experts/fuse_gate_alarm.py`;
  generation script (fresh mag=25 runs → 5var + diff CSV); a driver script.
- New split YAMLs (`*_gate25`, `*_test_ext25`, etc.) and batch_metrics configs.
- New result dirs (`ensemble_Gate_alarm_XMEAS10_*_test`); `metrics_alarmgate*.json`;
  clean-window summaries; a 3-way comparison table (A / B / new) on both test sets.
- Nothing existing modified; ±3σ and mag=100 limits unchanged.
