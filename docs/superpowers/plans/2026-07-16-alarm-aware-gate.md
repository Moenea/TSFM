# Alarm-aware Combined-score Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the XMEAS10/IDV13 gate the best mag=25 prognosis method (higher clean-window recall, lower false-prognosis rate, longer lead) by training it on an alarm-aware combined-score loss instead of MSE.

**Architecture:** Keep the softmax-over-4-experts gate + context features (add a temperature); replace the MSE training objective with a differentiable soft-alarm loss (recall + FAR + lead + small MSE regularizer). Feed it a fresh, disjoint mag=25 gate-training set so there is no expert leakage. All new files — nothing existing is edited.

**Tech Stack:** Python 3 (env `/home/aicode/miniconda3/envs/tsfm/bin/python`), PyTorch, NumPy, pandas, h5py (env `/home/aicode/miniconda3/envs/sherwin/bin/python` for h5 reads), the existing `scripts/adaptation/foundation_experts/common/expert_io.py` helpers.

## Global Constraints

- Repo root: `/home/aicode/sherwin/TSFM`. Data root: `/home/aicode/sherwin/dataset/TEP`.
- **±3σ is a hard threshold.** No tuned pre-alarm band. Prognosis event = forecast crosses ±3σ within horizon; true event = true crosses ±3σ.
- **Control limits fixed** from mag=100 pre-onset: LCL=0.1801943444436634, UCL=0.2437560483294926 (`setting/limits_tep_xmeas10.csv`). Gate `--train-split` for limits stays `setting/TEP_IDV13_XMEAS10.yaml` (mag=100, has `Time`).
- **Zero-risk:** do NOT edit `fuse_gate_multi.py`, `evaluate_multi.py`, `batch_metrics.py`, `evaluate.py`, `expert_io.py`, `_common.sh`, or any existing result dir / split. New files only.
- **Only mag=25 data.**
- seq_len=96, pred_len=96, horizon=15, half_start=7 (=15//2), dt=0.05 h, onset=30 h (row 600).
- Experts order everywhere: `diff, raw, time_moe, sundial` (N=4). Fused output contract: `pred.npy`/`true.npy` shape `(N_win, pred_len, 1)` float32.
- Fine-tune expert checkpoints reused (NOT retrained): exp-B mag=25 settings
  `forecast_TEP_IDV13_XMEAS10_5var_{raw,diff}_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0`.
- Python envs: `TSFM_PY=/home/aicode/miniconda3/envs/tsfm/bin/python`, `H5_PY=/home/aicode/miniconda3/envs/sherwin/bin/python`.
- Git commits are DEFERRED (user commits manually; repo is on detached HEAD). Do not run `git commit` unless the user asks.

---

## File Structure

- Create `scripts/adaptation/foundation_experts/gate_alarm/loss.py` — pure, testable alarm-aware loss + lead-weight + soft-alarm helpers.
- Create `scripts/adaptation/foundation_experts/gate_alarm/test_loss.py` — unit tests for `loss.py`.
- Create `scripts/adaptation/foundation_experts/fuse_gate_alarm.py` — CLI trainer/fuser (mirrors `fuse_gate_multi.py` I/O contract, new loss + args).
- Create `dataset/TEP/code/gen_lowmag_runs.py` — extract fresh mag=25 IDV13 runs from h5 → `csv_5var_lowmag/` + `csv_5var_lowmag_diff/`.
- Create split YAMLs under `setting/`: `TEP_IDV13_XMEAS10_5var_gate25.yaml` (+ `_diff_`), `..._val_gate25.yaml` per-fold, `..._testext25.yaml` (+ `_diff_`, + main).
- Create `scripts/adaptation/xmeas10/run_alarmgate.sh` — Phase 1 driver.
- Create `scripts/adaptation/xmeas10/select_alarmgate_cv.py` — grouped-CV hyperparameter sweep.
- Create batch_metrics configs `setting/batch_metrics_tep_xmeas10_{alarmgate_r910,alarmgate_ext,expA_ext,expB_ext}.yaml`.
- Create `scripts/adaptation/xmeas10/compare_3way.py` — final A/B/new comparison table.

---

## Task 1: Alarm-aware loss (pure functions + unit tests)

**Files:**
- Create: `scripts/adaptation/foundation_experts/gate_alarm/loss.py`
- Test: `scripts/adaptation/foundation_experts/gate_alarm/test_loss.py`

**Interfaces:**
- Produces:
  - `soft_alarm(fused, low, high, tau_a) -> Tensor` — per-step soft alarm `(B,H)`.
  - `window_soft_or(a, half_start) -> Tensor` — window soft-alarm `(B,)`.
  - `lead_weights(true_series, origins, low, high, horizon, dt, onset_h=30.0) -> np.ndarray` — per-window earliness weight in [0,1].
  - `alarm_aware_loss(fused, true, y_alarm, clean, lead_w, low, high, *, tau_a, lambda_far, lambda_lead, lambda_mse, half_start) -> (Tensor, dict)`.

- [ ] **Step 1: Write the failing tests**

```python
# scripts/adaptation/foundation_experts/gate_alarm/test_loss.py
import numpy as np
import torch
from loss import soft_alarm, window_soft_or, lead_weights, alarm_aware_loss

LOW, HIGH = 0.18, 0.244

def test_soft_alarm_high_when_above_limit():
    fused = torch.tensor([[0.30, 0.30]])          # well above HIGH
    a = soft_alarm(fused, LOW, HIGH, tau_a=0.003)
    assert float(a.min()) > 0.9

def test_soft_alarm_low_when_inside_band():
    fused = torch.tensor([[0.21, 0.21]])          # inside band
    a = soft_alarm(fused, LOW, HIGH, tau_a=0.003)
    assert float(a.max()) < 0.1

def test_window_soft_or_fires_if_any_latter_step_alarms():
    a = torch.tensor([[0.0, 0.0, 0.0, 0.95]])     # only last step alarms
    A = window_soft_or(a, half_start=2)
    assert float(A[0]) > 0.9

def test_window_soft_or_ignores_early_steps():
    a = torch.tensor([[0.95, 0.95, 0.0, 0.0]])    # only early steps alarm
    A = window_soft_or(a, half_start=2)
    assert float(A[0]) < 0.1

def test_lead_weights_earlier_origin_gets_more_weight():
    # crossing at index 120; window origins 100 (early) and 118 (late)
    s = np.full(200, 0.21); s[120:] = 0.30
    w = lead_weights(s, np.array([100, 118]), LOW, HIGH, horizon=15, dt=0.05)
    assert w[0] > w[1] >= 0.0 and w[0] <= 1.0

def test_loss_prefers_crossing_on_positives():
    # A positive window whose fused crosses -> lower loss than one that stays flat.
    y = torch.tensor([True]); clean = torch.tensor([True]); lead = torch.tensor([1.0])
    true = torch.full((1, 15), 0.30)
    cross = torch.full((1, 15), 0.30)             # fused crosses (matches true)
    flat  = torch.full((1, 15), 0.21)             # fused stays inside band
    kw = dict(low=LOW, high=HIGH, tau_a=0.003, lambda_far=1.0,
              lambda_lead=1.0, lambda_mse=0.1, half_start=7)
    l_cross, _ = alarm_aware_loss(cross, true, y, clean, lead, **kw)
    l_flat,  _ = alarm_aware_loss(flat,  true, y, clean, lead, **kw)
    assert float(l_cross) < float(l_flat)

def test_loss_penalizes_false_alarm_on_clean_negative():
    y = torch.tensor([False]); clean = torch.tensor([True]); lead = torch.tensor([0.0])
    true = torch.full((1, 15), 0.21)              # true stays inside band
    cross = torch.full((1, 15), 0.30); flat = torch.full((1, 15), 0.21)
    kw = dict(low=LOW, high=HIGH, tau_a=0.003, lambda_far=1.0,
              lambda_lead=1.0, lambda_mse=0.0, half_start=7)
    l_cross, _ = alarm_aware_loss(cross, true, y, clean, lead, **kw)
    l_flat,  _ = alarm_aware_loss(flat,  true, y, clean, lead, **kw)
    assert float(l_cross) > float(l_flat)         # crossing on a clean negative is worse
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/aicode/sherwin/TSFM/scripts/adaptation/foundation_experts/gate_alarm && /home/aicode/miniconda3/envs/tsfm/bin/python -m pytest test_loss.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'loss'` (file not yet created).

- [ ] **Step 3: Write the implementation**

```python
# scripts/adaptation/foundation_experts/gate_alarm/loss.py
"""Pure, testable alarm-aware combined-score loss for the XMEAS10 gate.
No I/O, no argparse — imported by fuse_gate_alarm.py and unit-tested standalone."""
from __future__ import annotations
import numpy as np
import torch

EPS = 1e-6

def soft_alarm(fused: torch.Tensor, low: float, high: float, tau_a: float) -> torch.Tensor:
    """Per-step differentiable ±3σ crossing indicator, (B,H) -> (B,H) in [0,1]."""
    a = torch.sigmoid((fused - high) / tau_a) + torch.sigmoid((low - fused) / tau_a)
    return a.clamp(0.0, 1.0)

def window_soft_or(a: torch.Tensor, half_start: int) -> torch.Tensor:
    """Soft-OR over the latter-half steps -> per-window alarm prob (B,)."""
    a_half = a[:, half_start:]
    return 1.0 - torch.prod(1.0 - a_half, dim=1)

def lead_weights(true_series: np.ndarray, origins: np.ndarray, low: float, high: float,
                 horizon: int, dt: float, onset_h: float = 30.0) -> np.ndarray:
    """Per-window earliness weight in [0,1]: for windows whose forecast origin
    precedes the run's first post-onset ±3σ crossing, weight rises the earlier the
    origin sits (normalized by the horizon length). Windows at/after the crossing
    or with no crossing get 0."""
    alarm = (true_series > high) | (true_series < low)
    idx = np.where(alarm)[0]
    idx = idx[idx * dt >= onset_h]
    if idx.size == 0:
        return np.zeros(len(origins), dtype=np.float64)
    cross = int(idx.min())
    lead = (cross - origins).astype(np.float64)           # steps ahead
    w = np.clip(lead / float(horizon), 0.0, 1.0)          # cap at one horizon
    w[origins >= cross] = 0.0
    return w

def alarm_aware_loss(fused, true, y_alarm, clean, lead_w, low, high, *,
                     tau_a, lambda_far, lambda_lead, lambda_mse, half_start):
    """fused,true: (B,H); y_alarm,clean: (B,) bool; lead_w: (B,) float in [0,1].
    Returns (scalar loss, component dict)."""
    a = soft_alarm(fused, low, high, tau_a)
    A = window_soft_or(a, half_start).clamp(EPS, 1.0 - EPS)      # (B,)
    y = y_alarm.float()
    pos = y_alarm
    neg = clean & (~y_alarm)
    elig = (pos | neg).float()
    bce = -(y * torch.log(A) + (1.0 - y) * torch.log(1.0 - A))
    bce = (bce * elig).sum() / elig.sum().clamp(min=1.0)
    negf = neg.float()
    far = (A * negf).sum() / negf.sum().clamp(min=1.0)
    posf = pos.float()
    lead = ((lead_w * A) * posf).sum() / posf.sum().clamp(min=1.0)
    mse = torch.mean((fused - true) ** 2)
    loss = bce + lambda_far * far - lambda_lead * lead + lambda_mse * mse
    return loss, {"bce": float(bce), "far": float(far), "lead": float(lead),
                  "mse": float(mse), "loss": float(loss)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/aicode/sherwin/TSFM/scripts/adaptation/foundation_experts/gate_alarm && /home/aicode/miniconda3/envs/tsfm/bin/python -m pytest test_loss.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5 (optional): commit** — only if the user asks. `git add scripts/adaptation/foundation_experts/gate_alarm/ && git commit -m "feat(gate): alarm-aware loss + tests"`

---

## Task 2: `fuse_gate_alarm.py` CLI trainer/fuser

**Files:**
- Create: `scripts/adaptation/foundation_experts/fuse_gate_alarm.py`
- Reference (read, do not edit): `scripts/adaptation/foundation_experts/fuse_gate_multi.py`, `common/expert_io.py`

**Interfaces:**
- Consumes: `gate_alarm.loss` (Task 1); `expert_io.control_limits/context_features/target_view/save_result`.
- Produces: a result dir with `pred.npy`,`true.npy`,`weights.npy`,`gate.pt`,`fit_log.json` (same contract as `fuse_gate_multi.py`), trained with the alarm-aware loss.
- CLI args: same as `fuse_gate_multi.py` (`--expert`,`--val-expert`,`--data-root`,`--train-split`,`--val-split`,`--test-split`,`--target`,`--output-dir`,`--seq-len`,`--pred-len`,`--horizon`,`--hidden`,`--epochs`,`--lr`,`--weight-decay`,`--seed`) PLUS `--tau-soft` (default 1.0), `--tau-a` (default 0.003), `--lambda-far` (default 1.0), `--lambda-lead` (default 1.0), `--lambda-mse` (default 0.1). `--val-*` here is the GATE-TRAINING data (the disjoint fresh runs), consistent with fuse_gate_multi's "train the gate on the val split" convention.

- [ ] **Step 1: Write the implementation** (adapted from `fuse_gate_multi.py`; changes: tempered softmax, alarm-aware loss, precompute y_alarm/clean/lead_w on the gate-training split)

```python
# scripts/adaptation/foundation_experts/fuse_gate_alarm.py
"""Alarm-aware combined-score gate. Same I/O contract as fuse_gate_multi.py, but
the gate is trained on soft-alarm recall/FAR/lead (not MSE), on a disjoint fresh
gate-training set (--val-*). ±3σ hard threshold; limits from --train-split."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "gate_alarm"))
from common import expert_io as io
from loss import alarm_aware_loss, lead_weights

class GateMLPMulti(nn.Module):
    def __init__(self, in_dim, hidden, horizon, n_experts, tau_soft):
        super().__init__()
        self.horizon, self.n_experts, self.tau_soft = horizon, n_experts, tau_soft
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, horizon * n_experts))
    def forward(self, x):
        logits = self.net(x).view(-1, self.horizon, self.n_experts) / self.tau_soft
        return torch.softmax(logits, dim=-1)

def fuse(weights, stack):
    return np.sum(weights * np.transpose(stack, (1, 2, 0)), axis=-1)

def _parse(items):
    out = []
    for it in items:
        name, _, d = it.partition(":")
        out.append((name, Path(d)))
    return out

def _load_stack(experts, horizon):
    preds, trues = [], []
    for _, d in experts:
        preds.append(io.target_view(np.load(d / "pred.npy"))[:, :horizon])
        trues.append(io.target_view(np.load(d / "true.npy"))[:, :horizon])
    return np.stack(preds, axis=0), trues

def _window_labels(data_root, split_path, target, seq_len, pred_len, horizon, low, high,
                   dt=0.05, onset_h=30.0):
    """Per-window y_alarm (latter-half true crosses), clean (30-step context clear),
    lead_w — concatenated across the split's test files, matching io.context_features order."""
    cfg = io.load_yaml(split_path)
    half = horizon // 2
    y_all, clean_all, lead_all = [], [], []
    for rel in cfg["test"]:
        s = io.read_target(data_root, rel, target)
        n = io.usable_count(len(s), seq_len, pred_len)
        origins = np.arange(n) + seq_len
        alarm_series = (s > high) | (s < low)
        for local in range(n):
            start = local + seq_len
            fut = s[start:start + horizon]
            fut_alarm = (fut > high) | (fut < low)
            y_all.append(bool(fut_alarm[half:].any()))
            clean_all.append(not alarm_series[max(0, start - 30):start].any())
        lead_all.append(lead_weights(s, origins, low, high, horizon, dt, onset_h))
    return (np.array(y_all), np.array(clean_all), np.concatenate(lead_all))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", action="append", required=True)
    p.add_argument("--val-expert", action="append", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-split", type=Path, required=True)
    p.add_argument("--val-split", type=Path, required=True)
    p.add_argument("--test-split", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau-soft", type=float, default=1.0)
    p.add_argument("--tau-a", type=float, default=0.003)
    p.add_argument("--lambda-far", type=float, default=1.0)
    p.add_argument("--lambda-lead", type=float, default=1.0)
    p.add_argument("--lambda-mse", type=float, default=0.1)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    test_experts, val_experts = _parse(a.expert), _parse(a.val_expert)
    n = len(test_experts)
    low, high = io.control_limits(a.data_root, a.train_split, a.target)
    val_stack, val_trues = _load_stack(val_experts, a.horizon)
    test_stack, _ = _load_stack(test_experts, a.horizon)
    val_true = val_trues[0]
    val_feat = io.context_features(a.data_root, a.val_split, a.target, a.seq_len, a.pred_len, low, high)
    test_feat = io.context_features(a.data_root, a.test_split, a.target, a.seq_len, a.pred_len, low, high)
    assert len(val_feat) == val_stack.shape[1]
    assert len(test_feat) == test_stack.shape[1]

    y_alarm, clean, lead_w = _window_labels(a.data_root, a.val_split, a.target,
                                            a.seq_len, a.pred_len, a.horizon, low, high)
    assert len(y_alarm) == val_stack.shape[1]

    mean = val_feat.mean(axis=0); scale = val_feat.std(axis=0) + 1e-6
    val_x = torch.tensor((val_feat - mean) / scale, dtype=torch.float32)
    test_x = torch.tensor((test_feat - mean) / scale, dtype=torch.float32)
    val_stack_t = torch.tensor(val_stack, dtype=torch.float32)
    val_y = torch.tensor(val_true, dtype=torch.float32)
    y_t = torch.tensor(y_alarm); clean_t = torch.tensor(clean)
    lead_t = torch.tensor(lead_w, dtype=torch.float32)

    gate = GateMLPMulti(8, a.hidden, a.horizon, n, a.tau_soft)
    opt = torch.optim.Adam(gate.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    logs = []
    for _ in range(a.epochs):
        opt.zero_grad()
        w = gate(val_x)                                       # (B,H,N)
        fused = torch.einsum("bhn,nbh->bh", w, val_stack_t)
        loss, comp = alarm_aware_loss(fused, val_y, y_t, clean_t, lead_t, low, high,
                                      tau_a=a.tau_a, lambda_far=a.lambda_far,
                                      lambda_lead=a.lambda_lead, lambda_mse=a.lambda_mse,
                                      half_start=a.horizon // 2)
        loss.backward(); opt.step(); logs.append(comp)

    gate.eval()
    with torch.no_grad():
        test_w = gate(test_x).numpy()
    fused_test = fuse(test_w, test_stack)
    base_full = io.target_view(np.load(test_experts[0][1] / "pred.npy"))
    out_full = base_full.copy(); out_full[:, :a.horizon] = fused_test
    io.save_result(a.output_dir, out_full,
                   io.target_view(np.load(test_experts[0][1] / "true.npy")),
                   {"method": "Gate-alarm", "experts": [nm for nm, _ in test_experts], "n_experts": n})
    np.save(a.output_dir / "weights.npy", test_w.astype(np.float32))
    torch.save(gate.state_dict(), a.output_dir / "gate.pt")
    (a.output_dir / "fit_log.json").write_text(json.dumps({
        "method": "Gate-alarm", "experts": [nm for nm, _ in test_experts],
        "final_components": logs[-1],
        "hparams": {"tau_soft": a.tau_soft, "tau_a": a.tau_a, "lambda_far": a.lambda_far,
                    "lambda_lead": a.lambda_lead, "lambda_mse": a.lambda_mse, "epochs": a.epochs},
        "mean_weight_by_expert": test_w.mean(axis=(0, 1)).tolist(),
        "control_limits_train_only": {"low": low, "high": high},
    }, indent=2))
    print(json.dumps({"final": logs[-1], "mean_w": test_w.mean(axis=(0, 1)).tolist()}, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI on existing exp-B data (Phase 0 dry run)**

Run (val = exp-B Run8 mag25 experts; test = mag25 Run9-10 = exp-B test dirs):
```bash
cd /home/aicode/sherwin/TSFM
DIFF=results/forecast_TEP_IDV13_XMEAS10_5var_diff_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAW=results/forecast_TEP_IDV13_XMEAS10_5var_raw_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
DIFFV=results/forecast_TEP_IDV13_XMEAS10_5var_diff_val_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAWV=results/forecast_TEP_IDV13_XMEAS10_5var_raw_val_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
/home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/foundation_experts/fuse_gate_alarm.py \
  --expert diff:$DIFF --expert raw:$RAW \
  --expert time_moe:results/fm_time_moe_xmeas10_mag25_test --expert sundial:results/fm_sundial_xmeas10_mag25_test \
  --val-expert diff:$DIFFV --val-expert raw:$RAWV \
  --val-expert time_moe:results/fm_time_moe_xmeas10_mag25_val --val-expert sundial:results/fm_sundial_xmeas10_mag25_val \
  --data-root /home/aicode/sherwin/dataset/TEP \
  --train-split setting/TEP_IDV13_XMEAS10.yaml \
  --val-split setting/TEP_IDV13_XMEAS10_5var_val_train25.yaml \
  --test-split setting/TEP_IDV13_XMEAS10_mag25.yaml \
  --target "XMEAS10 Purge Rate" --output-dir results/ensemble_Gate_alarm_XMEAS10_phase0_test \
  --lambda-far 1.0 --lambda-lead 1.0 --tau-soft 0.5 --tau-a 0.003 --epochs 800
```
Expected: prints `final` components + `mean_w` (4 numbers), writes the result dir with `pred.npy` shape `(3618, 96, 1)`.

- [ ] **Step 3: Verify shape + contract**

Run: `/home/aicode/miniconda3/envs/tsfm/bin/python -c "import numpy as np; p=np.load('results/ensemble_Gate_alarm_XMEAS10_phase0_test/pred.npy'); print(p.shape, p.dtype)"`
Expected: `(3618, 96, 1) float32`.

---

## Task 3: Phase 0 evaluation — does the loss beat exp A on existing data?

**Files:**
- Create: `setting/batch_metrics_tep_xmeas10_phase0.yaml`
- Reference: `utils/batch_metrics.py`, `results/XMEAS10 Purge Rate_Summary_mag25/summary.csv` (exp A clean-window).

- [ ] **Step 1: Write the batch_metrics config**

```yaml
# setting/batch_metrics_tep_xmeas10_phase0.yaml
params:
  target: XMEAS10 Purge Rate
  limit_csv_path: /home/aicode/sherwin/TSFM/setting/limits_tep_xmeas10.csv
  data_root: /home/aicode/sherwin/dataset/TEP/csv_5var_lowmag
  results_root: ./results
  seq_len: 96
  pred_len: 96
  eval_steps: 15
  input_clean_steps: 30
  alarm_quality_rmse_factor: 0.2
test:
- Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run9.csv
- Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run10.csv
model_dirs:
- {name: Gate-alarm-p0, result_dir: ensemble_Gate_alarm_XMEAS10_phase0_test}
```

- [ ] **Step 2: Run batch_metrics + compare to exp A**

```bash
cd /home/aicode/sherwin/TSFM
/home/aicode/miniconda3/envs/tsfm/bin/python -u utils/batch_metrics.py \
  --config setting/batch_metrics_tep_xmeas10_phase0.yaml --summary-suffix _phase0 --figure-suffix _phase0
/home/aicode/miniconda3/envs/tsfm/bin/python - <<'PY'
import pandas as pd
p0=pd.read_csv("results/XMEAS10 Purge Rate_Summary_phase0/summary.csv").iloc[0]
A =pd.read_csv("results/XMEAS10 Purge Rate_Summary_mag25/summary.csv")
A =A[A.model=="Gate"].iloc[0]
def show(tag,r): print(f"{tag}: recall={r.ratio_pred_in_true_alarm_patches_clean:.3f} "
      f"FAR={r.ratio_pred_in_no_true_alarm_patches_clean:.3f} lead={r.mean_lead_time_patch_clean*3:.2f}min")
show("exp A gate", A); show("alarm gate p0", p0)
PY
```
Expected: prints both rows. **Decision gate:** if alarm-gate p0 already ≥ exp A on recall AND ≤ on FAR AND ≥ on lead, proceed to Phase 1 with confidence. If not, sweep `--tau-soft {0.3,0.5,1.0}`, `--lambda-far {0.5,1,2}`, `--lambda-lead {0.5,1,2}` on this same Run8 setup first (cheap), and pick a promising region before Phase 1. Record the chosen ranges in `fit_log.json` notes.

---

## Task 4: Generate fresh mag=25 runs (gate + extended test)

**Files:**
- Create: `dataset/TEP/code/gen_lowmag_runs.py`

**Interfaces:**
- Produces: for each requested run N — `csv_5var_lowmag/...IDV13...13_25_Run{N}.csv` (5 cols, target last) and `csv_5var_lowmag_diff/...Run{N}.csv` (first-difference, row0=0).

- [ ] **Step 1: Write the generator**

```python
# dataset/TEP/code/gen_lowmag_runs.py
"""Extract fresh mag=25 IDV13 runs from TEP_mode1.h5 into 5-var raw + diff CSVs.
Usage: sherwin_python gen_lowmag_runs.py 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tep_loader import load_fault

COVS = ["XMEAS15 Stripper Level", "XMEAS17 Stripper Underflow",
        "XMV06 Purge", "XMV11 Condenser Coolant"]
TARGET = "XMEAS10 Purge Rate"
COLS = COVS + [TARGET]
BASE = Path("/home/aicode/sherwin/dataset/TEP")
V5 = BASE / "csv_5var_lowmag"; V5D = BASE / "csv_5var_lowmag_diff"
NAME = "Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run{n}.csv"

def main(runs):
    V5.mkdir(parents=True, exist_ok=True); V5D.mkdir(parents=True, exist_ok=True)
    for n in runs:
        r = load_fault(mode=1, idv=13, mag=25, run=int(n))
        df = r.proc.reset_index()            # brings 'Time' out; proc columns are named
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"missing {missing} in Run{n}; have {list(df.columns)[:8]}...")
        sub = df[COLS]
        sub.to_csv(V5 / NAME.format(n=n), index=False)
        vals = sub.to_numpy(dtype=np.float64)
        d = np.zeros_like(vals); d[1:] = np.diff(vals, axis=0)
        pd.DataFrame(d, columns=COLS).to_csv(V5D / NAME.format(n=n), index=False)
        print(f"Run{n}: wrote {len(sub)} rows -> {V5.name} + {V5D.name}")

if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 2: Generate the 17 fresh runs**

Run: `cd /home/aicode/sherwin/dataset/TEP/code && /home/aicode/miniconda3/envs/sherwin/bin/python gen_lowmag_runs.py 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27`
Expected: 17 lines `Run{n}: wrote 2000 rows -> csv_5var_lowmag + csv_5var_lowmag_diff` (2000 data rows).

- [ ] **Step 3: Verify counts + headers + diff convention**

Run:
```bash
cd /home/aicode/sherwin/dataset/TEP
for n in 11 17 27; do
  echo "Run$n raw_lines=$(wc -l < csv_5var_lowmag/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run${n}.csv) diff_lines=$(wc -l < csv_5var_lowmag_diff/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run${n}.csv)"
done
head -1 csv_5var_lowmag/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run11.csv
/home/aicode/miniconda3/envs/tsfm/bin/python -c "import pandas as pd; d=pd.read_csv('csv_5var_lowmag_diff/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run11.csv'); print('row0 all zero:', bool((d.iloc[0]==0).all()))"
```
Expected: each `raw_lines=2001 diff_lines=2001`; header = `XMEAS15 Stripper Level,XMEAS17 Stripper Underflow,XMV06 Purge,XMV11 Condenser Coolant,XMEAS10 Purge Rate`; `row0 all zero: True`.

---

## Task 5: Split YAMLs + leakage-free expert/zero-shot inference on fresh runs

**Files:**
- Create: `setting/TEP_IDV13_XMEAS10_5var_gate25.yaml`, `..._5var_diff_gate25.yaml` (test = Run11–17),
  `setting/TEP_IDV13_XMEAS10_5var_testext25.yaml`, `..._diff_testext25.yaml` (test = Run18–27),
  `setting/TEP_IDV13_XMEAS10_gate25.yaml`, `..._testext25.yaml` (main splits: train/val = mag100 csv/ for limits, test = fresh runs).
- Create: `scripts/adaptation/xmeas10/infer_freshruns.sh`
- Reference: `_common.sh` (`inference_args_ms`, `SUF`), `run_fm_zeroshot.sh` (zero-shot adapter calls).

**Interfaces:**
- Produces result dirs: `mag25_{diff,raw}_gate25`, `mag25_{diff,raw}_testext25`, `fm_{time_moe,sundial}_xmeas10_{gate25,testext25}`, plus exp-A (mag100) experts on extended test: `mag100_{diff,raw}_testext25`, `fm_{time_moe,sundial}_xmeas10_testext25` (zero-shot is magnitude-agnostic, shared).

- [ ] **Step 1: Write the split YAMLs** (generator script, run once)

```python
# scratchpad or inline: writes 6 split YAMLs
from pathlib import Path
import yaml
SET = Path("/home/aicode/sherwin/TSFM/setting"); TGT="XMEAS10 Purge Rate"
raw=lambda n:f"csv_5var_lowmag/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run{n}.csv"
dif=lambda n:f"csv_5var_lowmag_diff/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_25_Run{n}.csv"
main=lambda n:f"csv/Mode1_SingleFault_SimulationCompleted_IDV13_Mode1_IDVInfo_13_100_Run{n}.csv"   # mag100 for limits
FT=[main(n) for n in range(1,8)]; VAL=[main(8)]
gate=list(range(11,18)); ext=list(range(18,28))
def w(name, test): (SET/name).write_text("# fresh mag25 split (alarm-gate exp)\n"+yaml.safe_dump({"target":TGT,"train":FT,"val":VAL,"test":test},sort_keys=False))
w("TEP_IDV13_XMEAS10_5var_gate25.yaml",[raw(n) for n in gate])
w("TEP_IDV13_XMEAS10_5var_diff_gate25.yaml",[dif(n) for n in gate])
w("TEP_IDV13_XMEAS10_5var_testext25.yaml",[raw(n) for n in ext])
w("TEP_IDV13_XMEAS10_5var_diff_testext25.yaml",[dif(n) for n in ext])
w("TEP_IDV13_XMEAS10_gate25.yaml",[raw(n) for n in gate])
w("TEP_IDV13_XMEAS10_testext25.yaml",[raw(n) for n in ext])
print("wrote 6 splits")
```
Note: `train`/`val` point to mag=100 `csv/` (they carry `Time`, needed only by `control_limits`/`limits`); `test` points to the fresh mag=25 runs. Timer-XL raw/diff diff-restore reads `--raw_split_file` for its own raw test series — must match (raw split for raw inference, and the diff inference passes the corresponding raw gate/ext split).

- [ ] **Step 2: Write `infer_freshruns.sh`** (reuses exp-B Run1-7 checkpoints — NO training; zero-shot on fresh runs; mag100 exp-A experts on extended test)

```bash
#!/usr/bin/env bash
set -euo pipefail
export GPU_PHYSICAL=1
HERE=/home/aicode/sherwin/TSFM/scripts/adaptation/xmeas10
source "$HERE/_common.sh"
FE=$ROOT/scripts/adaptation/foundation_experts; R=results
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# exp-B (mag25) trained settings:
B_RAW=forecast_TEP_IDV13_XMEAS10_5var_raw_train25_few_r1p0_${SUF#timer_xl_}   # NOTE: SUF already full; see below
```
IMPORTANT — do not string-munge `SUF`. Use the literal exp-B setting names from Global Constraints. Concretely:
```bash
BSUF=timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
B_RAW=forecast_TEP_IDV13_XMEAS10_5var_raw_train25_few_r1p0_${BSUF}
B_DIFF=forecast_TEP_IDV13_XMEAS10_5var_diff_train25_few_r1p0_${BSUF}
A_RAW=forecast_TEP_IDV13_XMEAS10_5var_raw_few_r1p0_${BSUF}     # exp-A mag100 checkpoints
A_DIFF=forecast_TEP_IDV13_XMEAS10_5var_diff_few_r1p0_${BSUF}
TGT="XMEAS10 Purge Rate"

# ---- gate set (Run11-17): exp-B experts + zero-shot ----
inference_args_ms mag25_raw_gate25  $ROOT/setting/TEP_IDV13_XMEAS10_5var_gate25.yaml      "$B_RAW"  mag25_raw_gate25
inference_args_ms mag25_diff_gate25 $ROOT/setting/TEP_IDV13_XMEAS10_5var_diff_gate25.yaml "$B_DIFF" mag25_diff_gate25 \
  --restore_diff_to_raw --raw_split_file $ROOT/setting/TEP_IDV13_XMEAS10_5var_gate25.yaml --restore_target "$TGT"
for m in time_moe sundial; do
  extra=""; [ "$m" = sundial ] && extra="--num-samples 20 --batch-size 32"
  "$PYTHON_BIN" "$FE/$m/adapter.py" --mode predict --zero-shot $extra \
    --split-file $ROOT/setting/TEP_IDV13_XMEAS10_gate25.yaml --data-root "$DATA_ROOT" \
    --target "$TGT" --horizon 15 --out-dir "$R/fm_${m}_xmeas10_gate25" --device cuda:0
done

# ---- extended test (Run18-27): exp-B experts, exp-A experts, shared zero-shot ----
inference_args_ms mag25_raw_testext25  $ROOT/setting/TEP_IDV13_XMEAS10_5var_testext25.yaml      "$B_RAW"  mag25_raw_testext25
inference_args_ms mag25_diff_testext25 $ROOT/setting/TEP_IDV13_XMEAS10_5var_diff_testext25.yaml "$B_DIFF" mag25_diff_testext25 \
  --restore_diff_to_raw --raw_split_file $ROOT/setting/TEP_IDV13_XMEAS10_5var_testext25.yaml --restore_target "$TGT"
inference_args_ms mag100_raw_testext25  $ROOT/setting/TEP_IDV13_XMEAS10_5var_testext25.yaml      "$A_RAW"  mag100_raw_testext25
inference_args_ms mag100_diff_testext25 $ROOT/setting/TEP_IDV13_XMEAS10_5var_diff_testext25.yaml "$A_DIFF" mag100_diff_testext25 \
  --restore_diff_to_raw --raw_split_file $ROOT/setting/TEP_IDV13_XMEAS10_5var_testext25.yaml --restore_target "$TGT"
for m in time_moe sundial; do
  extra=""; [ "$m" = sundial ] && extra="--num-samples 20 --batch-size 32"
  "$PYTHON_BIN" "$FE/$m/adapter.py" --mode predict --zero-shot $extra \
    --split-file $ROOT/setting/TEP_IDV13_XMEAS10_testext25.yaml --data-root "$DATA_ROOT" \
    --target "$TGT" --horizon 15 --out-dir "$R/fm_${m}_xmeas10_testext25" --device cuda:0
done
echo "DONE fresh-run inference"
```

- [ ] **Step 3: Run it and verify shapes + leakage**

Run: `bash scripts/adaptation/xmeas10/infer_freshruns.sh`
Then verify:
```bash
/home/aicode/miniconda3/envs/tsfm/bin/python - <<'PY'
import numpy as np
for d,exp in [("mag25_diff_gate25",7*1809),("mag25_raw_gate25",7*1809),
              ("fm_time_moe_xmeas10_gate25",7*1809),("mag25_diff_testext25",10*1809),
              ("mag100_diff_testext25",10*1809),("fm_sundial_xmeas10_testext25",10*1809)]:
    n=np.load(f"results/{d}/pred.npy").shape[0]
    print(d, n, "OK" if n==exp else f"MISMATCH expected {exp}")
PY
```
Expected: gate dirs 12663 windows, ext dirs 18090; all `OK`.
**Leakage check:** exp-B/exp-A train splits are Run1–7 only; gate/ext runs are Run11–27 — disjoint by construction. Confirm the split files' `train:` lists contain no Run≥11.

---

## Task 6: Grouped-CV hyperparameter selection over the 7 gate runs

**Files:**
- Create: `scripts/adaptation/xmeas10/select_alarmgate_cv.py`
- Create per-fold split YAMLs on the fly (test = 6 gate runs train / 1 held out).

**Interfaces:**
- Consumes: gate-set expert dirs (Task 5), `fuse_gate_alarm.py` (Task 2).
- Produces: `results/TEP_IDV13_XMEAS10_Summary/alarmgate_cv.csv` (config × mean CV combined-S, recall, FAR, lead) and prints the winning config.

- [ ] **Step 1: Write the CV selector** (leave-one-gate-run-out; for each hparam config, fuse on 6 runs' windows, score clean combined-S on the held-out run using batch_metrics' clean definitions inline)

```python
# scripts/adaptation/xmeas10/select_alarmgate_cv.py
"""Grouped leave-one-run-out CV over the 7 gate runs to pick alarm-gate hparams.
Trains fuse_gate_alarm on 6 runs, scores clean recall/FAR/lead on the held-out run,
averages. Combined S = recall + (1 - FAR/0.05) + lead_steps/15. Prints ranked table."""
import itertools, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT=Path("/home/aicode/sherwin/TSFM"); PY="/home/aicode/miniconda3/envs/tsfm/bin/python"
DATA=Path("/home/aicode/sherwin/dataset/TEP"); TGT="XMEAS10 Purge Rate"
LOW,HIGH=0.1801943444436634,0.2437560483294926
GATE=list(range(11,18)); RAWDIR=DATA/"csv_5var_lowmag"
GRID=dict(tau_soft=[0.3,0.5,1.0], lambda_far=[0.5,1.0,2.0], lambda_lead=[0.5,1.0])
EXP={"diff":"mag25_diff_gate25","raw":"mag25_raw_gate25",
     "time_moe":"fm_time_moe_xmeas10_gate25","sundial":"fm_sundial_xmeas10_gate25"}
# ... (build per-fold splits, call fuse_gate_alarm with --val-split=6-run, --test-split=1-run,
#      slicing expert dirs by run via pred.npy window offsets 1809/run; compute clean recall/FAR/lead
#      on the held-out run; average; write alarmgate_cv.csv). Full body in repo.
```
Note: the selector reuses `batch_metrics`' clean definitions by importing the metric math, OR (simpler) shells out to `utils/batch_metrics.py` per fold with a generated config. Implementer picks one; the deliverable is `alarmgate_cv.csv` + printed winner. Keep it under ~60 min by limiting the grid to the 18 combos above.

- [ ] **Step 2: Run CV and pick the winner**

Run: `/home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/xmeas10/select_alarmgate_cv.py`
Expected: writes `results/TEP_IDV13_XMEAS10_Summary/alarmgate_cv.csv`, prints the top config by mean combined-S. Record `{tau_soft, lambda_far, lambda_lead}`.

---

## Task 7: Train final gate, evaluate on both test sets, 3-way comparison

**Files:**
- Create: `setting/batch_metrics_tep_xmeas10_{alarmgate_r910,alarmgate_ext,expB_ext,expA_ext}.yaml`
- Create: `scripts/adaptation/xmeas10/compare_3way.py`

- [ ] **Step 1: Train final gate on all 7 gate runs (winning hparams), test on Run9–10 and Run18–27**

```bash
cd /home/aicode/sherwin/TSFM
G=results
COMMON="--data-root /home/aicode/sherwin/dataset/TEP --train-split setting/TEP_IDV13_XMEAS10.yaml \
  --target 'XMEAS10 Purge Rate' --val-split setting/TEP_IDV13_XMEAS10_gate25.yaml \
  --val-expert diff:$G/mag25_diff_gate25 --val-expert raw:$G/mag25_raw_gate25 \
  --val-expert time_moe:$G/fm_time_moe_xmeas10_gate25 --val-expert sundial:$G/fm_sundial_xmeas10_gate25 \
  --lambda-far <WIN_FAR> --lambda-lead <WIN_LEAD> --tau-soft <WIN_TAU> --epochs 1500"
# Run9-10 (comparable):
eval $(echo /home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/foundation_experts/fuse_gate_alarm.py \
  --expert diff:results/mag25_diff_test --expert raw:results/mag25_raw_test \
  --expert time_moe:results/fm_time_moe_xmeas10_mag25_test --expert sundial:results/fm_sundial_xmeas10_mag25_test \
  --test-split setting/TEP_IDV13_XMEAS10_mag25.yaml --output-dir results/ensemble_Gate_alarm_XMEAS10_test $COMMON)
# Extended (Run18-27):
eval $(echo /home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/foundation_experts/fuse_gate_alarm.py \
  --expert diff:results/mag25_diff_testext25 --expert raw:results/mag25_raw_testext25 \
  --expert time_moe:results/fm_time_moe_xmeas10_testext25 --expert sundial:results/fm_sundial_xmeas10_testext25 \
  --test-split setting/TEP_IDV13_XMEAS10_testext25.yaml --output-dir results/ensemble_Gate_alarm_XMEAS10_testext $COMMON)
```
Replace `<WIN_*>` with Task 6 winners. Note the mag25_*_test dirs (Run9-10) are exp-B's existing test dirs — reused (same experts).

- [ ] **Step 2: Build exp-A and exp-B gates on the extended test** (for a fair 3-way comparison on Run18–27) using the EXISTING `fuse_gate_multi.py` (unchanged):

```bash
# exp-B gate (mag25 experts) on ext:
/home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/foundation_experts/fuse_gate_multi.py \
  --expert diff:results/mag25_diff_testext25 --expert raw:results/mag25_raw_testext25 \
  --expert time_moe:results/fm_time_moe_xmeas10_testext25 --expert sundial:results/fm_sundial_xmeas10_testext25 \
  --val-expert diff:results/forecast_TEP_IDV13_XMEAS10_5var_diff_val_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0 \
  --val-expert raw:results/forecast_TEP_IDV13_XMEAS10_5var_raw_val_train25_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0 \
  --val-expert time_moe:results/fm_time_moe_xmeas10_mag25_val --val-expert sundial:results/fm_sundial_xmeas10_mag25_val \
  --data-root /home/aicode/sherwin/dataset/TEP --train-split setting/TEP_IDV13_XMEAS10.yaml \
  --val-split setting/TEP_IDV13_XMEAS10_5var_val_train25.yaml --test-split setting/TEP_IDV13_XMEAS10_testext25.yaml \
  --target "XMEAS10 Purge Rate" --output-dir results/ensemble_Gate_expB_XMEAS10_testext
# exp-A gate (mag100 experts) on ext: same but val-experts = exp-A mag100 val dirs, test-experts = mag100_*_testext25 + shared zero-shot.
```

- [ ] **Step 3: batch_metrics on all gates for both test sets, then compare**

Write the four batch_metrics configs (test = Run9-10 or Run18-27; model_dirs = the gate dirs + 4 experts). Run each, then:
```bash
/home/aicode/miniconda3/envs/tsfm/bin/python scripts/adaptation/xmeas10/compare_3way.py
```
`compare_3way.py` loads the clean-window summaries for exp A / exp B / alarm gate on Run9-10 and on Run18-27, computes combined S, prints a table, and asserts the acceptance criterion (alarm gate recall ≥ exp A, FAR ≤ exp A, lead ≥ exp A on the extended test).
Expected: a printed 3-way table; PASS/FAIL line for the acceptance criterion.

- [ ] **Step 4 (optional): commit** — only if the user asks.

---

## Self-Review

**Spec coverage:**
- §2 constraints → Global Constraints (limits, ±3σ, zero-risk, mag25) ✓
- §4 architecture (softmax + temperature) → Task 2 `GateMLPMulti` ✓
- §5 loss → Task 1 (`loss.py` + tests) ✓
- §6 data (disjoint fresh runs) → Task 4 (gen) + Task 5 (splits/inference) ✓
- §7 model selection (grouped CV) → Task 6 ✓
- §8 evaluation (evaluate_multi + batch_metrics, both test sets) → Task 7 ✓
- §9 phases (Phase 0 sanity, Phase 1 full) → Task 3 decision gate + Tasks 4–7 ✓
- §3 combined score → Task 6 (CV) + Task 7 (compare_3way) ✓
- §10 fallbacks → not tasks (contingency); documented in spec ✓

**Placeholder scan:** Task 6 selector body and Task 7 exp-A ext gate are described rather than fully coded — these are mechanical (mirror Task 5/Task 2 patterns and the existing `fuse_gate_multi.py`); `<WIN_*>` are runtime values from Task 6. Acceptable as they depend on prior-task outputs; every novel/risky path (loss, gate, generator, inference) has complete code.

**Type consistency:** loss signatures in Task 1 tests match `loss.py`; `fuse_gate_alarm.py` imports `alarm_aware_loss`/`lead_weights` with matching kwargs; expert order `diff,raw,time_moe,sundial` consistent across all tasks; window counts 1809/run consistent.
