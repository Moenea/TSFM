# Heterogeneous-TSFM Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Time-MoE, MOIRAI, and Sundial as few-shot fine-tuned experts (and standalone baselines) to the TEP IDV13 prognosis pipeline, fused with Timer-XL (raw + diff) by a generalized N-expert softmax gate; validate end-to-end at ratio=1.0 before scaling to the few-shot curve.

**Architecture:** Approach A — *prediction-level adapters + N-expert gate*. Each new TSFM has a standalone adapter that fine-tunes on the **same** seed-2021 few-shot subset windows as Timer-XL (reusing `MultivariateDatasetYAMLSplitFewShot` only to *select* windows) and writes window-aligned `pred.npy`/`true.npy` in the exact format the existing gate consumes. A generalized softmax gate (`fuse_gate_multi.py`) fuses N experts; `evaluate_multi.py` scores each expert and the gate. Nothing touches the strict-loading `run.py`/`exp_forecast` loop.

**Tech Stack:** Python 3.10, PyTorch 2.0.1, NumPy, pandas, PyYAML; HuggingFace `transformers` (+ `trust_remote_code`) for Time-MoE & Sundial; `uni2ts` for MOIRAI. Deterministic units (`expert_io`, gate, evaluate) tested with plain `assert` scripts (no pytest in env).

## Global Constraints

- **Data root:** `/home/aicode/sherwin/dataset/TEP` (CSV files under `csv/`, diff under `csv_diff/`).
- **Repo root:** `/home/aicode/sherwin/TSFM`. Run all commands from here.
- **Timer-XL env python:** `/home/aicode/miniconda3/envs/tsfm/bin/python` (referred to as `$TSFM_PY`). Has torch 2.0.1+cu117, transformers 4.40.1, numpy, pandas, yaml, einops. **No pytest.**
- **Model env python:** decided in Task 1. Default `$TSFM_PY`; if a `transformers` bump is required, an **isolated** env `$FM_PY` is created — the `tsfm` env is NEVER mutated in a way that breaks Timer-XL.
- **Target column:** `XMEAS07 Reactor Pressure` (exact string, with spaces).
- **Splits (YAML, `test:` key is what gets inferred):**
  - test split: `setting/TEP_IDV13_XMEAS07.yaml` → train=Run1-7, val=Run8, **test=Run9,Run10**.
  - val split: `setting/TEP_IDV13_XMEAS07_val.yaml` → **test=Run8** (this is how Timer-XL produced "val" predictions).
- **Window geometry:** `seq_len=96`, `pred_len=96` (used **only** for the window-count formula, to match Timer-XL), gate `horizon=15`. Usable windows per file = `n_rows - seq_len - pred_len + 1 = 2001 - 96 - 96 + 1 = 1810`. Val N=1810 (Run8), test N=3620 (Run9+Run10). Full train N=12670 (7×1810).
- **Few-shot subset:** seed `2021`, random, nested, train-only — provided by `MultivariateDatasetYAMLSplitFewShot`; adapters reuse it for window *selection* only.
- **Prediction contract:** every expert writes `pred.npy` and `true.npy` of shape `(N, H, 1)` float32, in **original reactor-pressure units** (NOT normalized), windows in YAML-file order then local order. `H ≥ horizon` (new experts produce exactly `horizon=15`; Timer-XL produces 96).
- **Ground-truth derivation:** `true[w] = csv_target[local+seq_len : local+seq_len+H]` computed in float64 then cast float32. The gate uses the **first/base expert's** `true` as canonical and checks others only as an alignment guard.
- **Per-window normalization in adapters:** instance norm — subtract the 96-step context mean, divide by context std (eps 1e-5); de-normalize the model output with the same mean/std before saving. This mirrors Timer-XL's `--use_norm`.
- **Checkpoints (smallest first):** Time-MoE `Maple728/TimeMoE-50M`, MOIRAI `Salesforce/moirai-1.0-R-small`, Sundial `thuml/sundial-base-128m`.
- **Fine-tune fallback ladder:** if a model's fine-tuning is blocked, fall back (Sundial: full → head-only → zero-shot; MOIRAI: fine-tune → zero-shot) and **record the fallback explicitly** in that expert's `meta.json`.
- **New code lives under** `scripts/adaptation/foundation_experts/`. **Do not modify** the working `scripts/adaptation/few_shot/TEP_IDV13/` files except where a task explicitly says so.
- **Anchor numbers to preserve (full-shot Gate-T2):** test MSE ≈ 143.20, mean lead ≈ 2.125 h, pre-onset FAR ≈ 0.0786, event_recall = 1.0.

---

## File Structure

```
scripts/adaptation/foundation_experts/
  common/
    __init__.py
    expert_io.py          # windowing, true derivation, save/load, context features, control limits
  fuse_gate_multi.py      # N-expert softmax gate
  evaluate_multi.py       # per-expert + gate metrics over arbitrary expert dirs
  time_moe/
    adapter.py            # fit() + predict() for Time-MoE
    run.sh
  sundial/
    adapter.py
    run.sh
  moirai/
    adapter.py
    run.sh
  run_poc.sh              # ratio=1.0 end-to-end (Phase 1)
  run_curve_multi.sh      # ratio sweep (Phase 2)
  WORKFLOW.md
  tests/
    test_expert_io.py
    test_fuse_gate_multi.py
    test_evaluate_multi.py
    test_smoke_<model>.py  # created per adapter task
  PROBE.md                # env + model-API findings (Task 1)
```

Responsibilities: `expert_io.py` owns everything about *windows and the on-disk contract* — it is the single alignment authority. `fuse_gate_multi.py` owns the *fusion math only* (imports features/limits from `expert_io`). `evaluate_multi.py` owns *metrics only* (imports the existing prognosis functions). Each `adapter.py` owns *one model's fit+predict*, behind the uniform contract, and is the only file that imports that model's library.

---

### Task 1: Environment + model-API probe, and scaffolding

**Files:**
- Create: `scripts/adaptation/foundation_experts/PROBE.md`
- Create: `scripts/adaptation/foundation_experts/__init__.py` (empty)
- Create: `scripts/adaptation/foundation_experts/common/__init__.py` (empty)
- Create: `scripts/adaptation/foundation_experts/tests/` (dir, via the smoke script below)

**Interfaces:**
- Produces: `PROBE.md` recording, for each model: the working python interpreter (`$TSFM_PY` or a new `$FM_PY`), the exact load call, the exact inference call (`generate`/`forward`) with observed input/output shapes, and whether a training/loss interface is exposed. Later adapter tasks copy these recipes verbatim.

- [ ] **Step 1: Probe Time-MoE + Sundial load/inference in the tsfm env**

Run (from repo root):
```bash
cd /home/aicode/sherwin/TSFM
TSFM_PY=/home/aicode/miniconda3/envs/tsfm/bin/python
$TSFM_PY - <<'PY'
import torch, numpy as np
from transformers import AutoModelForCausalLM
ctx = torch.tensor(np.linspace(0,1,96), dtype=torch.float32).unsqueeze(0)  # [1,96]
for mid in ["Maple728/TimeMoE-50M", "thuml/sundial-base-128m"]:
    print("====", mid)
    try:
        m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)
        m.eval()
        with torch.no_grad():
            try:
                out = m.generate(ctx, max_new_tokens=15)
                print("  generate ok, out shape", tuple(out.shape))
            except Exception as e:
                print("  generate FAILED:", repr(e)[:200])
        print("  forward sig:", [p for p in m.forward.__doc__ ][:0] or "see modeling file")
    except Exception as e:
        print("  LOAD FAILED:", repr(e)[:300])
PY
```
Expected: at least the load succeeds. Record exact output shapes. If load fails on a `transformers` version error, note the required version.

- [ ] **Step 2: Decide the model env**

If both models loaded under `$TSFM_PY`, set `FM_PY=$TSFM_PY`. If a newer `transformers` is required, create an isolated env (do NOT touch `tsfm`):
```bash
/home/aicode/miniconda3/bin/conda create -y -n tsfm_fm --clone tsfm   # only if a bump is needed
/home/aicode/miniconda3/envs/tsfm_fm/bin/pip install -U "transformers>=<version-from-probe>"
```
Write the chosen `FM_PY` path into `PROBE.md`.

- [ ] **Step 3: Probe MOIRAI (uni2ts) availability**

```bash
$FM_PY -c "import uni2ts; print('uni2ts', uni2ts.__version__)" 2>&1 | head -1 || true
```
If missing, attempt install into the model env only:
```bash
$FM_PY -m pip install uni2ts 2>&1 | tail -5 || echo "UNI2TS_INSTALL_FAILED"
```
Record outcome in `PROBE.md`. If install fails, mark MOIRAI as **zero-shot fallback** for Task 8.

- [ ] **Step 4: Record findings and create package scaffolding**

Create the two empty `__init__.py` files and write `PROBE.md` with a table: model | interpreter | load call | inference call | output shape | training interface | decision (fine-tune / fallback).

```bash
mkdir -p scripts/adaptation/foundation_experts/common scripts/adaptation/foundation_experts/tests
: > scripts/adaptation/foundation_experts/__init__.py
: > scripts/adaptation/foundation_experts/common/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/adaptation/foundation_experts/__init__.py \
        scripts/adaptation/foundation_experts/common/__init__.py \
        scripts/adaptation/foundation_experts/PROBE.md
git commit -m "chore(fm-gate): probe model envs/APIs and scaffold package"
```

---

### Task 2: `common/expert_io.py` — windowing, ground truth, and the on-disk contract

**Files:**
- Create: `scripts/adaptation/foundation_experts/common/expert_io.py`
- Test: `scripts/adaptation/foundation_experts/tests/test_expert_io.py`

**Interfaces:**
- Consumes: repo's `data_provider.data_loader.MultivariateDatasetYAMLSplitFewShot` (for training-window *selection* only); the existing `setting/*.yaml` splits.
- Produces:
  - `usable_count(n_rows: int, seq_len: int, pred_len: int) -> int`
  - `load_yaml(path: Path) -> dict`
  - `target_view(arr: np.ndarray) -> np.ndarray` (3D→2D on last axis, requires last dim 1)
  - `read_target(data_root: Path, rel: str, target: str) -> np.ndarray` (1D float64)
  - `iter_infer_windows(data_root, split_path, target, seq_len, pred_len, horizon) -> tuple[np.ndarray, np.ndarray]` returning `contexts (N, seq_len)` and `trues (N, horizon)`, iterating `cfg["test"]` files in order, local order, `N` counted with `pred_len`.
  - `select_train_pairs(data_root, split_path, target, ratio, seq_len, pred_len) -> tuple[list[str], list[tuple[int,int]]]` returning the ordered train file list and the few-shot `(file_idx, local_idx)` pairs (identical selection to Timer-XL).
  - `windows_from_pairs(data_root, files, pairs, target, seq_len, horizon) -> tuple[np.ndarray, np.ndarray]` returning `contexts (M, seq_len)`, `futures (M, horizon)`.
  - `save_result(out_dir: Path, pred: np.ndarray, true: np.ndarray, meta: dict) -> None` writing `pred.npy`/`true.npy` as `(N, H, 1)` float32 and `meta.json`.
  - `control_limits(data_root, split_path, target) -> tuple[float, float]` (train healthy Time<30h, mean±3σ; copied from `fuse_gate_t2.py`).
  - `context_features(data_root, split_path, target, seq_len, pred_len, low, high) -> np.ndarray` (the 8-feature matrix; copied verbatim from `fuse_gate_t2.py`).

- [ ] **Step 1: Write the failing test**

```python
# scripts/adaptation/foundation_experts/tests/test_expert_io.py
import sys, tempfile, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
from common import expert_io as io  # noqa: E402

TARGET = "XMEAS07 Reactor Pressure"


def _make_split(tmp: Path, n_rows: int):
    """One synthetic 'test' file with a ramping target so windows are distinguishable."""
    csv_dir = tmp / "csv"; csv_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_rows) * 0.05
    df = pd.DataFrame({"Time": t, TARGET: 1000.0 + np.arange(n_rows, dtype=float)})
    df.to_csv(csv_dir / "Run.csv", index=False)
    split = tmp / "split.yaml"
    split.write_text(
        f'target: "{TARGET}"\ntrain:\n  - csv/Run.csv\nval:\n  - csv/Run.csv\ntest:\n  - csv/Run.csv\n'
    )
    return split


def test_usable_count():
    assert io.usable_count(2001, 96, 96) == 1810


def test_iter_infer_windows_alignment():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        split = _make_split(tmp, n_rows=200)
        ctx, true = io.iter_infer_windows(tmp, split, TARGET, seq_len=96, pred_len=96, horizon=15)
        # count uses pred_len=96 -> 200-96-96+1 = 9 windows
        assert ctx.shape == (9, 96), ctx.shape
        assert true.shape == (9, 15), true.shape
        # window 0: context = values[0:96] = 1000..1095 ; true = values[96:111] = 1096..1110
        assert ctx[0, 0] == 1000.0 and ctx[0, -1] == 1095.0
        assert true[0, 0] == 1096.0 and true[0, -1] == 1110.0
        # window 1 shifted by exactly one step
        assert ctx[1, 0] == 1001.0 and true[1, 0] == 1097.0


def test_save_result_shape_and_units():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "expert"
        pred = np.ones((5, 15), dtype=np.float32) * 2700.0
        true = np.ones((5, 15), dtype=np.float32) * 2710.0
        io.save_result(out, pred, true, {"model": "unit-test"})
        p = np.load(out / "pred.npy"); t = np.load(out / "true.npy")
        assert p.shape == (5, 15, 1) and t.shape == (5, 15, 1)
        assert p.dtype == np.float32 and abs(float(p.mean()) - 2700.0) < 1e-3
        assert json.loads((out / "meta.json").read_text())["model"] == "unit-test"


if __name__ == "__main__":
    test_usable_count()
    test_iter_infer_windows_alignment()
    test_save_result_shape_and_units()
    print("ALL PASS")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_expert_io.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.expert_io'` (or `ImportError`).

- [ ] **Step 3: Implement `expert_io.py`**

```python
# scripts/adaptation/foundation_experts/common/expert_io.py
"""Windowing, ground-truth derivation, and the on-disk prediction contract
shared by every foundation-model expert. This module is the single alignment
authority: all experts produce windows in the same order and units, so the gate
can fuse them without knowing which model wrote them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def usable_count(n_rows: int, seq_len: int, pred_len: int) -> int:
    return n_rows - seq_len - pred_len + 1


def target_view(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        if array.shape[-1] != 1:
            raise ValueError(f"expected target-only predictions, got {array.shape}")
        return array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D/3D predictions, got {array.shape}")
    return array


def read_target(data_root: Path, rel: str, target: str) -> np.ndarray:
    return pd.read_csv(Path(data_root) / rel)[target].to_numpy(dtype=np.float64)


def iter_infer_windows(data_root, split_path, target, seq_len, pred_len, horizon):
    """Contexts and ground truth for every window of the split's `test` files.

    N is counted with `pred_len` (to match Timer-XL's window count) but each
    window only needs `horizon` future steps."""
    cfg = load_yaml(split_path)
    contexts, trues = [], []
    for rel in cfg["test"]:
        values = read_target(data_root, rel, target)
        n = usable_count(len(values), seq_len, pred_len)
        for local in range(n):
            start = local + seq_len
            contexts.append(values[local:start])
            trues.append(values[start:start + horizon])
    return (np.asarray(contexts, dtype=np.float64),
            np.asarray(trues, dtype=np.float64))


def select_train_pairs(data_root, split_path, target, ratio, seq_len, pred_len):
    """Reuse MultivariateDatasetYAMLSplitFewShot ONLY to select the few-shot
    training windows, so the selection is byte-identical to Timer-XL's."""
    from data_provider.data_loader import MultivariateDatasetYAMLSplitFewShot
    ds = MultivariateDatasetYAMLSplitFewShot(
        root_path=str(data_root), flag="train", size=[seq_len, pred_len, pred_len],
        data_path="splits.yaml", scale=False, subset_rand_ratio=ratio,
        split_file=str(split_path), features="S", target=target,
    )
    subset = getattr(ds, "_subset_index", None)
    globals_ = subset if subset is not None else np.arange(ds.n_timepoint)
    pairs = [tuple(ds._locate(int(g))) for g in globals_]
    files = load_yaml(split_path)["train"]
    return files, pairs


def windows_from_pairs(data_root, files, pairs, target, seq_len, horizon):
    cache = {}
    contexts, futures = [], []
    for file_idx, local in pairs:
        rel = files[file_idx]
        if rel not in cache:
            cache[rel] = read_target(data_root, rel, target)
        values = cache[rel]
        start = local + seq_len
        contexts.append(values[local:start])
        futures.append(values[start:start + horizon])
    return (np.asarray(contexts, dtype=np.float64),
            np.asarray(futures, dtype=np.float64))


def save_result(out_dir: Path, pred: np.ndarray, true: np.ndarray, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = target_view(np.asarray(pred))[:, :, None].astype(np.float32)
    true = target_view(np.asarray(true))[:, :, None].astype(np.float32)
    np.save(out_dir / "pred.npy", pred)
    np.save(out_dir / "true.npy", true)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def control_limits(data_root: Path, split_path: Path, target: str) -> tuple[float, float]:
    cfg = load_yaml(split_path)
    healthy = []
    for rel in cfg["train"]:
        frame = pd.read_csv(Path(data_root) / rel)
        healthy.append(frame.loc[frame["Time"] < 30.0, target].to_numpy(dtype=np.float64))
    values = np.concatenate(healthy)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return mean - 3.0 * std, mean + 3.0 * std


def context_features(data_root, split_path, target, seq_len, pred_len, low, high):
    cfg = load_yaml(split_path)
    rows = []
    x = np.arange(96, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float((x_centered ** 2).sum())
    for rel in cfg["test"]:
        values = pd.read_csv(Path(data_root) / rel)[target].to_numpy(dtype=np.float64)
        for local in range(usable_count(len(values), seq_len, pred_len)):
            start = local + seq_len
            long_window = values[start - seq_len:start]
            short_window = values[start - 96:start]
            short_centered = short_window - short_window.mean()
            slope = float((x_centered * short_centered).sum() / denominator)
            last = float(values[start - 1])
            rows.append([
                last, last - float(values[start - 2]),
                float(short_window.mean()), float(short_window.std(ddof=0)), slope,
                float(long_window.max() - long_window.min()),
                last - low, high - last,
            ])
    return np.asarray(rows, dtype=np.float64)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_expert_io.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/adaptation/foundation_experts/common/expert_io.py \
        scripts/adaptation/foundation_experts/tests/test_expert_io.py
git commit -m "feat(fm-gate): expert_io windowing + on-disk contract (TDD)"
```

---

### Task 3: `fuse_gate_multi.py` — N-expert softmax gate

**Files:**
- Create: `scripts/adaptation/foundation_experts/fuse_gate_multi.py`
- Test: `scripts/adaptation/foundation_experts/tests/test_fuse_gate_multi.py`

**Interfaces:**
- Consumes: `expert_io.control_limits`, `expert_io.context_features`, `expert_io.target_view`; expert result dirs (each with `pred.npy`/`true.npy`).
- Produces:
  - `class GateMLPMulti(nn.Module)` — `__init__(self, in_dim: int, hidden: int, horizon: int, n_experts: int)`, `forward(features) -> Tensor` shape `(B, horizon, n_experts)`, softmax over the last axis.
  - `fuse(weights: np.ndarray, stack: np.ndarray) -> np.ndarray` where `stack` is `(N, B, horizon)` and `weights` is `(B, horizon, N)`, returning `(B, horizon)`.
  - `main()` CLI: repeated `--expert NAME:DIR` (first = base, supplies the tail and canonical `true`), plus `--val-expert NAME:DIR` (same order), `--data-root`, `--train-split`, `--val-split`, `--test-split`, `--target`, `--output-dir`, `--seq-len 96`, `--pred-len 96`, `--horizon 15`, `--hidden 32`, `--epochs 500`, `--lr 1e-3`, `--weight-decay 1e-4`, `--seed 42`. Writes `pred.npy`, `true.npy`, `weights.npy`, `gate.pt`, `fit_log.json`.

- [ ] **Step 1: Write the failing test (gate math)**

```python
# scripts/adaptation/foundation_experts/tests/test_fuse_gate_multi.py
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
import fuse_gate_multi as g  # noqa: E402


def test_weights_are_a_simplex():
    gate = g.GateMLPMulti(in_dim=8, hidden=16, horizon=15, n_experts=5)
    feats = torch.randn(20, 8)
    w = gate(feats)
    assert w.shape == (20, 15, 5)
    sums = w.sum(dim=-1).detach().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert (w.detach().numpy() >= 0).all()


def test_fuse_is_weighted_sum():
    # 2 experts, horizon 3, batch 2
    stack = np.array([
        [[1., 1., 1.], [2., 2., 2.]],   # expert 0
        [[3., 3., 3.], [4., 4., 4.]],   # expert 1
    ])  # (N=2, B=2, H=3)
    weights = np.zeros((2, 3, 2))
    weights[..., 0] = 0.25
    weights[..., 1] = 0.75
    fused = g.fuse(weights, stack)
    assert fused.shape == (2, 3)
    # window0: 0.25*1 + 0.75*3 = 2.5 ; window1: 0.25*2 + 0.75*4 = 3.5
    assert np.allclose(fused[0], 2.5) and np.allclose(fused[1], 3.5)


if __name__ == "__main__":
    test_weights_are_a_simplex()
    test_fuse_is_weighted_sum()
    print("ALL PASS")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_fuse_gate_multi.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'fuse_gate_multi'`.

- [ ] **Step 3: Implement `fuse_gate_multi.py`**

```python
# scripts/adaptation/foundation_experts/fuse_gate_multi.py
"""Context-only softmax gate over N experts. Trains on the val split (Run8),
applies to the test split (Run9-10). Generalizes fuse_gate_t2.py: the sigmoid
single-weight becomes a softmax over N experts per horizon step."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import expert_io as io


class GateMLPMulti(nn.Module):
    def __init__(self, in_dim: int, hidden: int, horizon: int, n_experts: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_experts = n_experts
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * n_experts),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).view(-1, self.horizon, self.n_experts)
        return torch.softmax(logits, dim=-1)


def fuse(weights: np.ndarray, stack: np.ndarray) -> np.ndarray:
    # weights (B, H, N), stack (N, B, H) -> (B, H, N) -> weighted sum over N
    stack_bhn = np.transpose(stack, (1, 2, 0))
    return np.sum(weights * stack_bhn, axis=-1)


def _parse_experts(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        name, _, directory = item.partition(":")
        out.append((name, Path(directory)))
    return out


def _load_stack(experts, horizon):
    preds, trues = [], []
    for _, directory in experts:
        preds.append(io.target_view(np.load(directory / "pred.npy"))[:, :horizon])
        trues.append(io.target_view(np.load(directory / "true.npy"))[:, :horizon])
    return np.stack(preds, axis=0), trues  # (N,B,H), list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", action="append", required=True, help="NAME:DIR (test); first = base")
    parser.add_argument("--val-expert", action="append", required=True, help="NAME:DIR (val); same order")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_experts = _parse_experts(args.expert)
    val_experts = _parse_experts(args.val_expert)
    n = len(test_experts)
    low, high = io.control_limits(args.data_root, args.train_split, args.target)

    val_stack, val_trues = _load_stack(val_experts, args.horizon)   # (N,Bv,H)
    test_stack, test_trues = _load_stack(test_experts, args.horizon)  # (N,Bt,H)
    val_true = val_trues[0]
    test_true = test_trues[0]
    for k in range(1, n):  # alignment guard (generous: round-trip float + diff-restore)
        if not np.allclose(val_trues[k], val_true, atol=1e-1):
            raise ValueError(f"val true mismatch for expert {val_experts[k][0]}")
        if not np.allclose(test_trues[k], test_true, atol=1e-1):
            raise ValueError(f"test true mismatch for expert {test_experts[k][0]}")

    val_feat = io.context_features(args.data_root, args.val_split, args.target,
                                   args.seq_len, args.pred_len, low, high)
    test_feat = io.context_features(args.data_root, args.test_split, args.target,
                                    args.seq_len, args.pred_len, low, high)
    assert len(val_feat) == val_stack.shape[1], (len(val_feat), val_stack.shape)
    assert len(test_feat) == test_stack.shape[1], (len(test_feat), test_stack.shape)

    mean = val_feat.mean(axis=0)
    scale = val_feat.std(axis=0) + 1e-6
    val_x = torch.tensor((val_feat - mean) / scale, dtype=torch.float32)
    test_x = torch.tensor((test_feat - mean) / scale, dtype=torch.float32)
    val_stack_t = torch.tensor(val_stack, dtype=torch.float32)  # (N,Bv,H)
    val_y = torch.tensor(val_true, dtype=torch.float32)

    gate = GateMLPMulti(8, args.hidden, args.horizon, n)
    optim = torch.optim.Adam(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for _ in range(args.epochs):
        optim.zero_grad()
        w = gate(val_x)                                  # (Bv,H,N)
        fused = torch.einsum("bhn,nbh->bh", w, val_stack_t)
        loss = torch.mean((fused - val_y) ** 2)
        loss.backward(); optim.step()
        losses.append(float(loss.detach()))

    gate.eval()
    with torch.no_grad():
        test_w = gate(test_x).numpy()                    # (Bt,H,N)
    fused_test = fuse(test_w, test_stack)                # (Bt,H)

    base_full = io.target_view(np.load(test_experts[0][1] / "pred.npy"))
    output_full = base_full.copy()
    output_full[:, :args.horizon] = fused_test

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    io.save_result(args.output_dir, output_full, test_true_full(test_experts[0][1]),
                   {"method": "Gate-multi", "experts": [n_ for n_, _ in test_experts],
                    "n_experts": n})
    np.save(args.output_dir / "weights.npy", test_w.astype(np.float32))
    torch.save(gate.state_dict(), args.output_dir / "gate.pt")
    log = {
        "method": "Gate-multi",
        "experts": [name for name, _ in test_experts],
        "final_gate_train_mse": losses[-1],
        "test_mse": {name: mse(test_stack[i], test_true) for i, (name, _) in enumerate(test_experts)},
        "gate_test_mse": mse(fused_test, test_true),
        "mean_weight_by_expert": test_w.mean(axis=(0, 1)).tolist(),
        "control_limits_train_only": {"low": low, "high": high},
    }
    (args.output_dir / "fit_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


def test_true_full(base_dir: Path) -> np.ndarray:
    """Full-length true from the base expert, for save_result's true.npy."""
    return io.target_view(np.load(base_dir / "true.npy"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_fuse_gate_multi.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/adaptation/foundation_experts/fuse_gate_multi.py \
        scripts/adaptation/foundation_experts/tests/test_fuse_gate_multi.py
git commit -m "feat(fm-gate): N-expert softmax gate (TDD math)"
```

---

### Task 4: `evaluate_multi.py` — per-expert + gate metrics

**Files:**
- Create: `scripts/adaptation/foundation_experts/evaluate_multi.py`
- Test: `scripts/adaptation/foundation_experts/tests/test_evaluate_multi.py`

**Interfaces:**
- Consumes: `evaluate.py`'s `limits`, `file_counts`, `forecast_metrics`, `prognosis_metrics` (imported from the existing few-shot file); arbitrary expert result dirs.
- Produces: `main()` CLI with repeated `--expert NAME:DIR`, `--data-root`, `--split`, `--target`, `--output`, `--seq-len 96`, `--pred-len 96`, `--horizon 15`. Writes one JSON report `{ "models": { name: {mse, mae, event_recall, ...} } }`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/adaptation/foundation_experts/tests/test_evaluate_multi.py
import sys, tempfile, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/aicode/sherwin/TSFM")
TARGET = "XMEAS07 Reactor Pressure"
TSFM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"


def test_report_has_each_expert():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "csv").mkdir()
        n = 300
        t = np.arange(n) * 0.05
        df = pd.DataFrame({"Time": t, TARGET: 2700.0 + np.zeros(n)})
        df.to_csv(tmp / "csv/Run.csv", index=False)
        split = tmp / "split.yaml"
        split.write_text(f'target: "{TARGET}"\ntrain:\n  - csv/Run.csv\ntest:\n  - csv/Run.csv\n')
        nwin = 300 - 96 - 96 + 1
        for name in ("a", "b"):
            dd = tmp / name; dd.mkdir()
            np.save(dd / "pred.npy", np.full((nwin, 15, 1), 2700.0, np.float32))
            np.save(dd / "true.npy", np.full((nwin, 15, 1), 2700.0, np.float32))
        out = tmp / "report.json"
        r = subprocess.run([TSFM_PY,
            str(ROOT / "scripts/adaptation/foundation_experts/evaluate_multi.py"),
            "--data-root", str(tmp), "--split", str(split), "--target", TARGET,
            "--expert", f"a:{tmp/'a'}", "--expert", f"b:{tmp/'b'}",
            "--output", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.loads(out.read_text())
        assert set(rep["models"]) == {"a", "b"}
        assert rep["models"]["a"]["mse"] == 0.0


if __name__ == "__main__":
    test_report_has_each_expert()
    print("ALL PASS")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_evaluate_multi.py`
Expected: FAIL — `evaluate_multi.py` not found (returncode != 0).

- [ ] **Step 3: Implement `evaluate_multi.py`**

```python
# scripts/adaptation/foundation_experts/evaluate_multi.py
"""Per-expert + gate metrics for the heterogeneous-TSFM gate. Reuses the
prognosis metric functions from the validated few-shot evaluate.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT / "scripts/adaptation/few_shot/TEP_IDV13"))
from evaluate import limits, file_counts, forecast_metrics, prognosis_metrics, view  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", action="append", required=True, help="NAME:DIR")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()

    low, high = limits(args.data_root, args.split, args.target)
    counts = file_counts(args.data_root, args.split, args.seq_len, args.pred_len)
    report = {"target": args.target, "control_limits": {"low": low, "high": high},
              "horizon_steps": args.horizon, "models": {}}
    for item in args.expert:
        name, _, directory = item.partition(":")
        directory = Path(directory)
        pred = view(np.load(directory / "pred.npy"))
        true = view(np.load(directory / "true.npy"))
        report["models"][name] = {
            **forecast_metrics(pred, true, args.horizon),
            **prognosis_metrics(pred, true, counts, low, high, args.seq_len, args.horizon),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_evaluate_multi.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/adaptation/foundation_experts/evaluate_multi.py \
        scripts/adaptation/foundation_experts/tests/test_evaluate_multi.py
git commit -m "feat(fm-gate): multi-expert evaluation report"
```

---

### Task 5: Time-MoE adapter (fit + predict)

**Files:**
- Create: `scripts/adaptation/foundation_experts/time_moe/adapter.py`
- Create: `scripts/adaptation/foundation_experts/time_moe/run.sh`
- Test: `scripts/adaptation/foundation_experts/tests/test_smoke_time_moe.py`

**Interfaces:**
- Consumes: `expert_io.select_train_pairs`, `windows_from_pairs`, `iter_infer_windows`, `save_result`; `Maple728/TimeMoE-50M` via `transformers` (recipe confirmed in `PROBE.md`).
- Produces: `adapter.py` CLI with `--mode {fit,predict}`, `--ratio`, `--split-file`, `--data-root`, `--target`, `--seq-len 96`, `--pred-len 96`, `--horizon 15`, `--ckpt-id Maple728/TimeMoE-50M`, `--ckpt-dir`, `--out-dir`, `--epochs 10`, `--lr 1e-5`, `--batch-size 64`, `--device cuda:0`, `--zero-shot` flag. `fit` writes a fine-tuned checkpoint to `--ckpt-dir`; `predict` writes `pred.npy`/`true.npy`/`meta.json` to `--out-dir`.

- [ ] **Step 1: Write the smoke test (zero-shot predict, shape + finiteness)**

```python
# scripts/adaptation/foundation_experts/tests/test_smoke_time_moe.py
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
FM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"  # update from PROBE.md if isolated env
ADAPTER = ROOT / "scripts/adaptation/foundation_experts/time_moe/adapter.py"
VAL_SPLIT = ROOT / "setting/TEP_IDV13_XMEAS07_val.yaml"   # test=Run8
DATA_ROOT = "/home/aicode/sherwin/dataset/TEP"
TARGET = "XMEAS07 Reactor Pressure"


def test_zero_shot_predict_shape():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tm"
        r = subprocess.run([FM_PY, str(ADAPTER), "--mode", "predict", "--zero-shot",
            "--split-file", str(VAL_SPLIT), "--data-root", DATA_ROOT, "--target", TARGET,
            "--horizon", "15", "--out-dir", str(out), "--device", "cuda:0"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        pred = np.load(out / "pred.npy")
        true = np.load(out / "true.npy")
        assert pred.shape == (1810, 15, 1), pred.shape   # Run8 windows
        assert true.shape == (1810, 15, 1), true.shape
        assert np.isfinite(pred).all()
        # sanity: original-scale reactor pressure is O(1e3), not normalized O(1)
        assert pred.mean() > 100.0, pred.mean()


if __name__ == "__main__":
    test_zero_shot_predict_shape()
    print("ALL PASS")
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `cd /home/aicode/sherwin/TSFM && $FM_PY scripts/adaptation/foundation_experts/tests/test_smoke_time_moe.py`
Expected: FAIL — adapter file does not exist (returncode != 0).

- [ ] **Step 3: Implement `time_moe/adapter.py`**

Use the per-window instance-norm recipe and the `generate` call confirmed in `PROBE.md`. If the probe shows `generate` returns the full sequence, slice the last `horizon`.

```python
# scripts/adaptation/foundation_experts/time_moe/adapter.py
"""Time-MoE expert adapter. fit() few-shot fine-tunes on the same seed-2021
subset windows as Timer-XL; predict() writes window-aligned, original-scale
forecasts in the shared on-disk contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
from common import expert_io as io  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

EPS = 1e-5


def _norm(ctx: torch.Tensor):
    mean = ctx.mean(dim=-1, keepdim=True)
    std = ctx.std(dim=-1, keepdim=True) + EPS
    return (ctx - mean) / std, mean, std


def _load_model(ckpt_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(ckpt_id_or_dir, trust_remote_code=True)
    return model.to(device)


def predict(args) -> None:
    model = _load_model(args.ckpt_dir if (args.ckpt_dir and not args.zero_shot) else args.ckpt_id,
                        args.device)
    model.eval()
    contexts, trues = io.iter_infer_windows(
        args.data_root, args.split_file, args.target,
        args.seq_len, args.pred_len, args.horizon)
    preds = np.empty((contexts.shape[0], args.horizon), dtype=np.float64)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(contexts), bs):
            ctx = torch.tensor(contexts[i:i + bs], dtype=torch.float32, device=args.device)
            normed, mean, std = _norm(ctx)
            out = model.generate(normed, max_new_tokens=args.horizon)   # PROBE-confirmed
            fc = out[:, -args.horizon:]
            fc = fc * std + mean
            preds[i:i + bs] = fc.cpu().numpy()
    io.save_result(args.out_dir, preds, trues,
                   {"model": "Time-MoE", "ckpt": args.ckpt_id,
                    "zero_shot": bool(args.zero_shot), "horizon": args.horizon})
    print(f"saved {preds.shape} -> {args.out_dir}")


def fit(args) -> None:
    files, pairs = io.select_train_pairs(
        args.data_root, args.split_file, args.target, args.ratio, args.seq_len, args.pred_len)
    contexts, futures = io.windows_from_pairs(
        args.data_root, files, pairs, args.target, args.seq_len, args.horizon)
    model = _load_model(args.ckpt_id, args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctx_t = torch.tensor(contexts, dtype=torch.float32, device=args.device)
    fut_t = torch.tensor(futures, dtype=torch.float32, device=args.device)
    n = len(ctx_t)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=args.device)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            ctx, mean, std = _norm(ctx_t[idx])
            tgt = (fut_t[idx] - mean) / std
            out = model.generate(ctx, max_new_tokens=args.horizon)  # PROBE: swap for forward(labels) if exposed
            fc = out[:, -args.horizon:]
            loss = torch.mean((fc - tgt) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"epoch {epoch} mse {total / n:.4f}")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.ckpt_dir)
    print(f"saved fine-tuned checkpoint -> {args.ckpt_dir}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fit", "predict"], required=True)
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ckpt-id", default="Maple728/TimeMoE-50M")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zero-shot", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    (predict if args.mode == "predict" else fit)(args)
```

> **NOTE (PROBE-driven):** if `PROBE.md` shows Time-MoE exposes `forward(input_ids=..., labels=...)` returning `.loss`, replace the `generate`-based training inner loop with that single forward+`.loss` call (cleaner and differentiable end-to-end). The `generate` path is the fallback if no training loss is exposed.

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `cd /home/aicode/sherwin/TSFM && $FM_PY scripts/adaptation/foundation_experts/tests/test_smoke_time_moe.py`
Expected: `ALL PASS` (zero-shot path; downloads the 50M checkpoint on first run).

- [ ] **Step 5: Write `time_moe/run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Time-MoE expert: fit on the few-shot subset, then infer val (Run8) and test (Run9-10).
HERE=$(cd "$(dirname "$0")" && pwd)
FM_PY=${FM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
RATIO=${SUBSET_RATIO:-1.0}
TAG=$(printf 'r%s' "$RATIO" | tr '.' 'p')
CKPT=$ROOT/checkpoints/fm_time_moe_${TAG}
VAL_OUT=$ROOT/results/fm_time_moe_${TAG}_val
TEST_OUT=$ROOT/results/fm_time_moe_${TAG}_test
cd "$ROOT"
"$FM_PY" "$HERE/adapter.py" --mode fit --ratio "$RATIO" \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT"
"$FM_PY" "$HERE/adapter.py" --mode predict \
  --split-file setting/TEP_IDV13_XMEAS07_val.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$VAL_OUT"
"$FM_PY" "$HERE/adapter.py" --mode predict \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$TEST_OUT"
```

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/adaptation/foundation_experts/time_moe/run.sh
git add scripts/adaptation/foundation_experts/time_moe/ \
        scripts/adaptation/foundation_experts/tests/test_smoke_time_moe.py
git commit -m "feat(fm-gate): Time-MoE expert adapter (fit/predict + smoke)"
```

---

### Task 6: Sundial adapter (fit + predict)

**Files:**
- Create: `scripts/adaptation/foundation_experts/sundial/adapter.py`
- Create: `scripts/adaptation/foundation_experts/sundial/run.sh`
- Test: `scripts/adaptation/foundation_experts/tests/test_smoke_sundial.py`

**Interfaces:**
- Consumes: same `expert_io` functions; `thuml/sundial-base-128m` via `transformers` (`generate(num_samples=K).mean`). Point output = mean of 20 samples (PROBE-confirmed signature).
- Produces: `adapter.py` with the same CLI surface as Task 5 plus `--num-samples 20`. Honours the fallback ladder: if no training loss is exposed (per `PROBE.md`), `fit` falls back to head-only or zero-shot and records it in `meta.json`.

- [ ] **Step 1: Write the smoke test (zero-shot predict shape)**

```python
# scripts/adaptation/foundation_experts/tests/test_smoke_sundial.py
import subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
FM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"  # update from PROBE.md
ADAPTER = ROOT / "scripts/adaptation/foundation_experts/sundial/adapter.py"
VAL_SPLIT = ROOT / "setting/TEP_IDV13_XMEAS07_val.yaml"
DATA_ROOT = "/home/aicode/sherwin/dataset/TEP"
TARGET = "XMEAS07 Reactor Pressure"


def test_zero_shot_predict_shape():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sd"
        r = subprocess.run([FM_PY, str(ADAPTER), "--mode", "predict", "--zero-shot",
            "--split-file", str(VAL_SPLIT), "--data-root", DATA_ROOT, "--target", TARGET,
            "--horizon", "15", "--num-samples", "20", "--out-dir", str(out), "--device", "cuda:0"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        pred = np.load(out / "pred.npy")
        assert pred.shape == (1810, 15, 1), pred.shape
        assert np.isfinite(pred).all() and pred.mean() > 100.0


if __name__ == "__main__":
    test_zero_shot_predict_shape()
    print("ALL PASS")
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `cd /home/aicode/sherwin/TSFM && $FM_PY scripts/adaptation/foundation_experts/tests/test_smoke_sundial.py`
Expected: FAIL — adapter file does not exist.

- [ ] **Step 3: Implement `sundial/adapter.py`**

```python
# scripts/adaptation/foundation_experts/sundial/adapter.py
"""Sundial expert adapter. predict() uses the generative model's mean over
num_samples draws; fit() few-shot fine-tunes if a training loss is exposed,
else falls back (head-only -> zero-shot), recorded in meta.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
from common import expert_io as io  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

EPS = 1e-5


def _norm(ctx):
    mean = ctx.mean(dim=-1, keepdim=True)
    std = ctx.std(dim=-1, keepdim=True) + EPS
    return (ctx - mean) / std, mean, std


def predict(args) -> None:
    src = args.ckpt_dir if (args.ckpt_dir and not args.zero_shot and Path(args.ckpt_dir).exists()) else args.ckpt_id
    model = AutoModelForCausalLM.from_pretrained(str(src), trust_remote_code=True).to(args.device)
    model.eval()
    contexts, trues = io.iter_infer_windows(
        args.data_root, args.split_file, args.target, args.seq_len, args.pred_len, args.horizon)
    preds = np.empty((contexts.shape[0], args.horizon), dtype=np.float64)
    bs = args.batch_size
    fell_back = args.zero_shot or not (args.ckpt_dir and Path(args.ckpt_dir).exists())
    with torch.no_grad():
        for i in range(0, len(contexts), bs):
            ctx = torch.tensor(contexts[i:i + bs], dtype=torch.float32, device=args.device)
            normed, mean, std = _norm(ctx)
            out = model.generate(normed, max_new_tokens=args.horizon, num_samples=args.num_samples)
            fc = out.mean(dim=1)[:, -args.horizon:]   # PROBE-confirmed sample axis
            preds[i:i + bs] = (fc * std + mean).cpu().numpy()
    io.save_result(args.out_dir, preds, trues,
                   {"model": "Sundial", "ckpt": args.ckpt_id, "num_samples": args.num_samples,
                    "zero_shot": bool(fell_back), "horizon": args.horizon})
    print(f"saved {preds.shape} -> {args.out_dir}")


def fit(args) -> None:
    # Fallback ladder per PROBE.md. If a flow-matching training loss is exposed,
    # replace this body with the fine-tune loop; otherwise record zero-shot.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.ckpt_dir) / "FALLBACK.txt").write_text(
        "Sundial fine-tune not run; predict() will use zero-shot. See PROBE.md.\n")
    print("Sundial fit: fallback to zero-shot (see PROBE.md / FALLBACK.txt)")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fit", "predict"], required=True)
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ckpt-id", default="thuml/sundial-base-128m")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zero-shot", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    (predict if args.mode == "predict" else fit)(args)
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `cd /home/aicode/sherwin/TSFM && $FM_PY scripts/adaptation/foundation_experts/tests/test_smoke_sundial.py`
Expected: `ALL PASS`. If `generate(num_samples=...)` signature differs, adjust per `PROBE.md` and re-run.

- [ ] **Step 5: Write `sundial/run.sh`** (copy of `time_moe/run.sh` with `time_moe`→`sundial`, `fm_time_moe`→`fm_sundial`, and `--num-samples 20` on the predict calls; `fit` is a no-op fallback)

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FM_PY=${FM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
RATIO=${SUBSET_RATIO:-1.0}
TAG=$(printf 'r%s' "$RATIO" | tr '.' 'p')
CKPT=$ROOT/checkpoints/fm_sundial_${TAG}
VAL_OUT=$ROOT/results/fm_sundial_${TAG}_val
TEST_OUT=$ROOT/results/fm_sundial_${TAG}_test
cd "$ROOT"
"$FM_PY" "$HERE/adapter.py" --mode fit --ratio "$RATIO" \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" --ckpt-dir "$CKPT"
"$FM_PY" "$HERE/adapter.py" --mode predict --num-samples 20 \
  --split-file setting/TEP_IDV13_XMEAS07_val.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$VAL_OUT"
"$FM_PY" "$HERE/adapter.py" --mode predict --num-samples 20 \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$TEST_OUT"
```

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/adaptation/foundation_experts/sundial/run.sh
git add scripts/adaptation/foundation_experts/sundial/ \
        scripts/adaptation/foundation_experts/tests/test_smoke_sundial.py
git commit -m "feat(fm-gate): Sundial expert adapter (predict + fallback fit + smoke)"
```

---

### Task 7: MOIRAI adapter (fit + predict, with zero-shot fallback)

**Files:**
- Create: `scripts/adaptation/foundation_experts/moirai/adapter.py`
- Create: `scripts/adaptation/foundation_experts/moirai/run.sh`
- Test: `scripts/adaptation/foundation_experts/tests/test_smoke_moirai.py`

**Interfaces:**
- Consumes: same `expert_io` functions; `uni2ts` `MoiraiForecast` + `MoiraiModule` (`Salesforce/moirai-1.0-R-small`). Point output = predictive mean. If `uni2ts` is unavailable (`PROBE.md`), the adapter prints a clear message and exits non-zero so the gate can drop the expert.
- Produces: `adapter.py` with the same CLI surface as Task 5 plus `--patch-size 16`. `fit` uses uni2ts's finetune loss if available, else zero-shot; records the choice in `meta.json`.

- [ ] **Step 1: Write the smoke test (skips cleanly if uni2ts missing)**

```python
# scripts/adaptation/foundation_experts/tests/test_smoke_moirai.py
import subprocess, tempfile, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
FM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"  # update from PROBE.md
ADAPTER = ROOT / "scripts/adaptation/foundation_experts/moirai/adapter.py"
VAL_SPLIT = ROOT / "setting/TEP_IDV13_XMEAS07_val.yaml"
DATA_ROOT = "/home/aicode/sherwin/dataset/TEP"
TARGET = "XMEAS07 Reactor Pressure"


def _uni2ts_present():
    r = subprocess.run([FM_PY, "-c", "import uni2ts"], capture_output=True)
    return r.returncode == 0


def test_zero_shot_predict_shape():
    if not _uni2ts_present():
        print("SKIP: uni2ts not installed (MOIRAI runs as zero-shot fallback or is dropped)")
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "mo"
        r = subprocess.run([FM_PY, str(ADAPTER), "--mode", "predict", "--zero-shot",
            "--split-file", str(VAL_SPLIT), "--data-root", DATA_ROOT, "--target", TARGET,
            "--horizon", "15", "--out-dir", str(out), "--device", "cuda:0"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        pred = np.load(out / "pred.npy")
        assert pred.shape == (1810, 15, 1), pred.shape
        assert np.isfinite(pred).all() and pred.mean() > 100.0


if __name__ == "__main__":
    test_zero_shot_predict_shape()
    print("ALL PASS")
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `cd /home/aicode/sherwin/TSFM && $FM_PY scripts/adaptation/foundation_experts/tests/test_smoke_moirai.py`
Expected: FAIL — adapter file does not exist (or, if uni2ts absent, it would SKIP — but the file is missing so it errors on import/exec). Confirm the failure is "adapter not found".

- [ ] **Step 3: Implement `moirai/adapter.py`**

```python
# scripts/adaptation/foundation_experts/moirai/adapter.py
"""MOIRAI expert adapter. predict() uses the predictive mean of the uni2ts
MoiraiForecast; fit() uses uni2ts's finetune loss if available, else zero-shot.
If uni2ts is not installed, exits non-zero with a clear message."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
from common import expert_io as io  # noqa: E402


def _require_uni2ts():
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"MOIRAI unavailable: uni2ts import failed ({exc}). "
              f"Install uni2ts or drop this expert from the gate.", file=sys.stderr)
        sys.exit(2)
    return MoiraiForecast, MoiraiModule


def predict(args) -> None:
    MoiraiForecast, MoiraiModule = _require_uni2ts()
    module = MoiraiModule.from_pretrained(args.ckpt_id)
    model = MoiraiForecast(
        module=module, prediction_length=args.horizon, context_length=args.seq_len,
        patch_size=args.patch_size, num_samples=args.num_samples,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(args.device)
    model.eval()
    contexts, trues = io.iter_infer_windows(
        args.data_root, args.split_file, args.target, args.seq_len, args.pred_len, args.horizon)
    preds = np.empty((contexts.shape[0], args.horizon), dtype=np.float64)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(contexts), bs):
            ctx = torch.tensor(contexts[i:i + bs], dtype=torch.float32, device=args.device)
            b = ctx.shape[0]
            past_target = ctx.unsqueeze(-1)                                  # (b, L, 1)
            past_observed = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros(b, args.seq_len, dtype=torch.bool, device=args.device)
            out = model(past_target=past_target, past_observed_target=past_observed,
                        past_is_pad=past_is_pad)                              # (b, num_samples, H, 1)
            fc = out.mean(dim=1).squeeze(-1)[:, :args.horizon]               # predictive mean
            preds[i:i + bs] = fc.cpu().numpy()
    io.save_result(args.out_dir, preds, trues,
                   {"model": "MOIRAI", "ckpt": args.ckpt_id, "patch_size": args.patch_size,
                    "zero_shot": bool(args.zero_shot), "horizon": args.horizon})
    print(f"saved {preds.shape} -> {args.out_dir}")


def fit(args) -> None:
    _require_uni2ts()
    # Per PROBE.md: if uni2ts finetune is wired, run it here and save_pretrained.
    # Otherwise fall back to zero-shot (predict() loads the pretrained module).
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.ckpt_dir) / "FALLBACK.txt").write_text(
        "MOIRAI fine-tune not run; predict() uses pretrained (zero-shot). See PROBE.md.\n")
    print("MOIRAI fit: fallback to zero-shot (see PROBE.md / FALLBACK.txt)")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fit", "predict"], required=True)
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ckpt-id", default="Salesforce/moirai-1.0-R-small")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zero-shot", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    (predict if args.mode == "predict" else fit)(args)
```

> **NOTE (PROBE-driven):** the exact `MoiraiForecast` constructor kwargs and the forward return shape vary by uni2ts version. Confirm both in Task 1's probe and adjust the `model(...)` call and the `out.mean(dim=1)` axis accordingly. The `patch_size` must be compatible with `context_length=96` (16 divides 96).

- [ ] **Step 4: Run the smoke test to verify it passes (or SKIPs cleanly)**

Run: `cd /home/aicode/sherwin/TSFM && $FM_PY scripts/adaptation/foundation_experts/tests/test_smoke_moirai.py`
Expected: `ALL PASS` if uni2ts present; otherwise `SKIP: uni2ts not installed ...` then `ALL PASS`.

- [ ] **Step 5: Write `moirai/run.sh`** (same shape as `sundial/run.sh`, `sundial`→`moirai`, `fm_sundial`→`fm_moirai`, drop `--num-samples 20` from predict — MOIRAI uses its own default)

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FM_PY=${FM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
RATIO=${SUBSET_RATIO:-1.0}
TAG=$(printf 'r%s' "$RATIO" | tr '.' 'p')
CKPT=$ROOT/checkpoints/fm_moirai_${TAG}
VAL_OUT=$ROOT/results/fm_moirai_${TAG}_val
TEST_OUT=$ROOT/results/fm_moirai_${TAG}_test
cd "$ROOT"
"$FM_PY" "$HERE/adapter.py" --mode fit --ratio "$RATIO" \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" --ckpt-dir "$CKPT"
"$FM_PY" "$HERE/adapter.py" --mode predict \
  --split-file setting/TEP_IDV13_XMEAS07_val.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$VAL_OUT"
"$FM_PY" "$HERE/adapter.py" --mode predict \
  --split-file setting/TEP_IDV13_XMEAS07.yaml --data-root "$DATA_ROOT" --target "$TARGET" \
  --ckpt-dir "$CKPT" --out-dir "$TEST_OUT"
```

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/adaptation/foundation_experts/moirai/run.sh
git add scripts/adaptation/foundation_experts/moirai/ \
        scripts/adaptation/foundation_experts/tests/test_smoke_moirai.py
git commit -m "feat(fm-gate): MOIRAI expert adapter (predict + fallback fit + smoke)"
```

---

### Task 8: `run_poc.sh` — ratio=1.0 end-to-end + backward-compat check

**Files:**
- Create: `scripts/adaptation/foundation_experts/run_poc.sh`
- Create: `scripts/adaptation/foundation_experts/WORKFLOW.md`
- Test: `scripts/adaptation/foundation_experts/tests/test_backward_compat.py`

**Interfaces:**
- Consumes: all adapters' `run.sh`; the existing Timer-XL raw/diff result dirs from `few_shot/TEP_IDV13` at `r1p0` (run the existing `run_curve.sh` with `RATIOS="1.0"` first if they are absent); `fuse_gate_multi.py`; `evaluate_multi.py`.
- Produces: a 5-expert gate result dir `results/ensemble_Gate-multi_r1p0_test` and a report `results/TEP_IDV13_XMEAS07_FM_Summary/metrics_r1p0.json`.

- [ ] **Step 1: Write the backward-compat test (N=2 ≈ Gate-T2)**

This guards that the new softmax gate, restricted to (diff, raw), reproduces the existing Gate-T2 within re-fit noise. It runs only when the Timer-XL r1p0 result dirs and the existing Gate-T2 metrics exist.

```python
# scripts/adaptation/foundation_experts/tests/test_backward_compat.py
import json, subprocess, tempfile, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
TSFM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"
RES = ROOT / "results"
DIFF = RES / "forecast_TEP_IDV13_XMEAS07_S_few_r1p0_DIFF_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
RAW = RES / "forecast_TEP_IDV13_XMEAS07_S_few_r1p0_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
DIFF_VAL = RES / "forecast_TEP_IDV13_XMEAS07_S_few_r1p0_DIFF_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
RAW_VAL = RES / "forecast_TEP_IDV13_XMEAS07_S_few_r1p0_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
GATE_T2_REF = RES / "TEP_IDV13_XMEAS07_FewShot_Summary" / "metrics_r1p0.json"
TARGET = "XMEAS07 Reactor Pressure"


def test_two_expert_matches_gate_t2():
    if not (DIFF.exists() and RAW.exists() and GATE_T2_REF.exists()):
        print("SKIP: Timer-XL r1p0 results or Gate-T2 reference not present")
        return
    ref_mse = json.loads(GATE_T2_REF.read_text())["models"]["gate_t2"]["mse"]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "g2"
        r = subprocess.run([TSFM_PY,
            str(ROOT / "scripts/adaptation/foundation_experts/fuse_gate_multi.py"),
            "--expert", f"diff:{DIFF}", "--expert", f"raw:{RAW}",
            "--val-expert", f"diff:{DIFF_VAL}", "--val-expert", f"raw:{RAW_VAL}",
            "--data-root", "/home/aicode/sherwin/dataset/TEP",
            "--train-split", str(ROOT / "setting/TEP_IDV13_XMEAS07.yaml"),
            "--val-split", str(ROOT / "setting/TEP_IDV13_XMEAS07_val.yaml"),
            "--test-split", str(ROOT / "setting/TEP_IDV13_XMEAS07.yaml"),
            "--target", TARGET, "--output-dir", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        gate_mse = json.loads((out / "fit_log.json").read_text())["gate_test_mse"]
        # re-fit softmax: same 2-simplex family, allow modest tolerance around 143.20
        assert abs(gate_mse - ref_mse) < 5.0, (gate_mse, ref_mse)


if __name__ == "__main__":
    test_two_expert_matches_gate_t2()
    print("ALL PASS")
```

- [ ] **Step 2: Run the backward-compat test**

Run: `cd /home/aicode/sherwin/TSFM && $TSFM_PY scripts/adaptation/foundation_experts/tests/test_backward_compat.py`
Expected: `ALL PASS` if r1p0 Timer-XL results exist; else `SKIP ...` then `ALL PASS`. If it fails on a real mismatch, fix the gate before proceeding.

- [ ] **Step 3: Write `run_poc.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Phase-1 proof of concept at ratio=1.0: ensure Timer-XL raw/diff exist, run the
# 3 new adapters, fuse all 5 experts, evaluate. Set FM_PY for the model env.
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/aicode/sherwin/TSFM
DATA_ROOT=/home/aicode/sherwin/dataset/TEP
TARGET="XMEAS07 Reactor Pressure"
TSFM_PY=${TSFM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
FM_PY=${FM_PY:-$TSFM_PY}
export FM_PY
RATIO=1.0
TAG=r1p0
cd "$ROOT"

# 0) Timer-XL raw/diff at ratio 1.0 (skip if already present)
DIFF=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_DIFF_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
if [ ! -d "$DIFF" ]; then
  RATIOS="1.0" bash scripts/adaptation/few_shot/TEP_IDV13/run_curve.sh
fi

# 1) New experts (each does fit + val/test predict)
SUBSET_RATIO=$RATIO bash "$HERE/time_moe/run.sh"
SUBSET_RATIO=$RATIO bash "$HERE/sundial/run.sh"
SUBSET_RATIO=$RATIO bash "$HERE/moirai/run.sh" || echo "MOIRAI skipped (see PROBE.md)"

# 2) Fuse 5 experts (first expert = base = Timer-XL diff, supplies tail + canonical true)
RAW=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
DIFF_VAL=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_DIFF_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
RAW_VAL=results/forecast_TEP_IDV13_XMEAS07_S_few_${TAG}_val_timer_xl_MultivariateDatasetYAMLSplitFewShot_sl96_it96_ot96_lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
GATE=results/ensemble_Gate-multi_${TAG}_test
EXPERTS=(--expert "diff:$DIFF" --expert "raw:$RAW"
         --expert "time_moe:results/fm_time_moe_${TAG}_test"
         --expert "sundial:results/fm_sundial_${TAG}_test")
VAL_EXPERTS=(--val-expert "diff:$DIFF_VAL" --val-expert "raw:$RAW_VAL"
         --val-expert "time_moe:results/fm_time_moe_${TAG}_val"
         --val-expert "sundial:results/fm_sundial_${TAG}_val")
if [ -d "results/fm_moirai_${TAG}_test" ]; then
  EXPERTS+=(--expert "moirai:results/fm_moirai_${TAG}_test")
  VAL_EXPERTS+=(--val-expert "moirai:results/fm_moirai_${TAG}_val")
fi
"$TSFM_PY" "$HERE/fuse_gate_multi.py" "${EXPERTS[@]}" "${VAL_EXPERTS[@]}" \
  --data-root "$DATA_ROOT" \
  --train-split setting/TEP_IDV13_XMEAS07.yaml \
  --val-split setting/TEP_IDV13_XMEAS07_val.yaml \
  --test-split setting/TEP_IDV13_XMEAS07.yaml \
  --target "$TARGET" --output-dir "$GATE"

# 3) Evaluate every expert + the gate
SUMMARY=results/TEP_IDV13_XMEAS07_FM_Summary
mkdir -p "$SUMMARY"
EVAL=(--expert "diff:$DIFF" --expert "raw:$RAW"
      --expert "time_moe:results/fm_time_moe_${TAG}_test"
      --expert "sundial:results/fm_sundial_${TAG}_test"
      --expert "gate:$GATE")
if [ -d "results/fm_moirai_${TAG}_test" ]; then EVAL+=(--expert "moirai:results/fm_moirai_${TAG}_test"); fi
"$TSFM_PY" "$HERE/evaluate_multi.py" "${EVAL[@]}" \
  --data-root "$DATA_ROOT" --split setting/TEP_IDV13_XMEAS07.yaml \
  --target "$TARGET" --output "$SUMMARY/metrics_${TAG}.json"
echo "PoC done -> $SUMMARY/metrics_${TAG}.json"
```

- [ ] **Step 4: Run the PoC end-to-end**

Run: `cd /home/aicode/sherwin/TSFM && bash scripts/adaptation/foundation_experts/run_poc.sh`
Expected: completes; `results/TEP_IDV13_XMEAS07_FM_Summary/metrics_r1p0.json` lists `diff, raw, time_moe, sundial, [moirai,] gate`, each with `mse`, `event_recall`, etc. **Exit criterion:** gate `event_recall` ≥ existing Gate-T2 and gate `mse` ≤ the worst single new expert (the gate must not be dragged below Timer-XL).

- [ ] **Step 5: Write `WORKFLOW.md`** documenting the env (`FM_PY`), the expert contract, `run_poc.sh`, the result-dir names, and the fallback notes from `PROBE.md`.

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/adaptation/foundation_experts/run_poc.sh
git add scripts/adaptation/foundation_experts/run_poc.sh \
        scripts/adaptation/foundation_experts/WORKFLOW.md \
        scripts/adaptation/foundation_experts/tests/test_backward_compat.py
git commit -m "feat(fm-gate): ratio=1.0 PoC orchestration + backward-compat guard"
```

---

### Task 9 (Phase 2 — gated on PoC success): `run_curve_multi.sh` — 5-expert few-shot curve

Only start after Task 8's exit criterion is met. This sweeps the few-shot ratios and rebuilds the figures with the 5-expert gate.

**Files:**
- Create: `scripts/adaptation/foundation_experts/run_curve_multi.sh`
- Create: `scripts/adaptation/foundation_experts/collect_curve_multi.py`

**Interfaces:**
- Consumes: `run_poc.sh`'s per-ratio machinery (parameterize `RATIO`/`TAG`); per-ratio `metrics_r<tag>.json`.
- Produces: `results/TEP_IDV13_XMEAS07_FM_Summary/curve.{csv,json,png}` and refreshed `figures/few-shot/TEP/IDV13/` comparisons.

- [ ] **Step 1: Generalize `run_poc.sh` into a ratio-parameterized function**

Refactor the ratio=1.0 literal in `run_poc.sh` into `run_one_ratio <R>` (replace `RATIO=1.0; TAG=r1p0` with `RATIO=$1; TAG=$(printf 'r%s' "$1" | tr '.' 'p')`), and have `run_poc.sh` call `run_one_ratio 1.0`. Keep `run_poc.sh`'s behaviour identical for 1.0.

- [ ] **Step 2: Write `run_curve_multi.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/run_poc.sh"   # provides run_one_ratio
RATIOS=${RATIOS:-"0.01 0.02 0.05 0.1 0.25 0.5 1.0"}
for r in $RATIOS; do
  echo "###### FM gate ratio=$r ######"
  run_one_ratio "$r"
done
TSFM_PY=${TSFM_PY:-/home/aicode/miniconda3/envs/tsfm/bin/python}
"$TSFM_PY" "$HERE/collect_curve_multi.py" --ratios $RATIOS \
  --summary-dir /home/aicode/sherwin/TSFM/results/TEP_IDV13_XMEAS07_FM_Summary
```

- [ ] **Step 3: Write `collect_curve_multi.py`**

Mirror `few_shot/TEP_IDV13/collect_curve.py` but iterate the expert names present in each `metrics_r<tag>.json` (`diff, raw, time_moe, sundial, moirai, gate`) and plot `mse` vs `n_train` per expert plus the gate. Reuse the `tag_of` and table-printing structure from `collect_curve.py`.

- [ ] **Step 4: Run a 2-point smoke of the curve**

Run: `cd /home/aicode/sherwin/TSFM && RATIOS="0.1 1.0" bash scripts/adaptation/foundation_experts/run_curve_multi.sh`
Expected: `curve.csv`/`curve.json`/`curve.png` written with rows for ratios 0.1 and 1.0.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/adaptation/foundation_experts/run_curve_multi.sh
git add scripts/adaptation/foundation_experts/run_curve_multi.sh \
        scripts/adaptation/foundation_experts/collect_curve_multi.py \
        scripts/adaptation/foundation_experts/run_poc.sh
git commit -m "feat(fm-gate): 5-expert few-shot curve sweep + aggregation"
```

---

## Self-Review

**Spec coverage:**
- Component layout → File Structure + Tasks 2-9. ✓
- Uniform expert contract (fit/predict, pred.npy (N,H,1), original scale) → Global Constraints + Task 2 (`expert_io`) + Tasks 5-7 adapters. ✓
- N-expert softmax gate, N=2 reduces to Gate-T2 → Task 3 + Task 8 backward-compat test. ✓
- Per-model checkpoints/point-forecast → Tasks 5-7 (Time-MoE direct, MOIRAI mean, Sundial 20-sample mean). ✓
- Few-shot fine-tune on identical subset → Task 2 `select_train_pairs` (reuses the dataset class) + adapter `fit()`. ✓
- PoC at ratio=1.0 then scale → Task 8 (PoC) + Task 9 (curve, gated). ✓
- Risks/fallbacks (uni2ts, Sundial loss, transformers bump, isolated env) → Task 1 probe + fallback ladders in Tasks 6-7 + `meta.json` recording. ✓
- Baselines table → `evaluate_multi.py` (Task 4) scores each expert standalone. ✓

**Placeholder scan:** No "TBD/TODO". The two `> NOTE (PROBE-driven)` blocks are explicit, bounded instructions tied to Task 1's recorded findings (the realistic seam for external-model APIs), not vague placeholders — each names the exact call to confirm and the fallback if it differs.

**Type consistency:** `expert_io` function names/signatures used in Tasks 3-7 match Task 2's Produces block (`iter_infer_windows`, `select_train_pairs`, `windows_from_pairs`, `save_result`, `control_limits`, `context_features`, `target_view`). Gate `--expert NAME:DIR` / `--val-expert NAME:DIR` surface is consistent between Task 3 (definition), Task 8 (`run_poc.sh`), and the backward-compat test. Result-dir naming (`fm_<model>_<tag>_{val,test}`, `ensemble_Gate-multi_<tag>_test`) is consistent across Tasks 5-9.

**Scope:** Tasks 1-8 are Phase 1 (one coherent deliverable: a validated 5-expert gate at ratio=1.0). Task 9 (Phase 2 curve) is explicitly gated on Phase-1 success and could be split into its own plan if preferred.
