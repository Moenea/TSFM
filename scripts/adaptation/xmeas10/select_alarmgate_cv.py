"""Grouped leave-one-run-out CV over the 7 gate runs (Run11-17) to pick the
alarm-gate hyperparameters. Trains fuse_gate_alarm's model on 6 runs, predicts the
held-out run; pools all 7 held-out predictions and scores clean recall/FAR/lead with
a scorer ported to match utils/batch_metrics.py exactly. Combined score
S = recall + (1 - FAR/0.05) + lead_steps/15.

Run `--validate` first: reproduces the phase-0 gate's Run9-10 clean metrics
(recall 0.491, FAR 0.024, lead 4.65 min) to prove the scorer matches batch_metrics.
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts/gate_alarm"))
from common import expert_io as io  # noqa: E402
from loss import alarm_aware_loss  # noqa: E402
from fuse_gate_alarm import GateMLPMulti, _window_labels, _limits_from_csv  # noqa: E402

DATA = Path("/home/aicode/sherwin/dataset/TEP")
TGT = "XMEAS10 Purge Rate"
SEQ, PRED, H = 96, 96, 15
EXPERTS = ["diff", "raw", "time_moe", "sundial"]
LOW, HIGH = _limits_from_csv(ROOT / "setting/limits_tep_xmeas10_mag25.csv", TGT)


def contiguous_events(mask):
    events, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            events.append((start, i - 1)); start = None
    if start is not None:
        events.append((start, len(mask) - 1))
    return events


def build_window_starts(seq_len, pred_len, file_lengths):
    usable = [max(0, L - seq_len - pred_len + 1) for L in file_lengths]
    cum = np.cumsum([0] + usable)
    offsets = np.cumsum([0] + list(file_lengths))
    total = cum[-1]
    starts = np.zeros(total, dtype=np.int64)
    for g in range(total):
        fi = int(np.searchsorted(cum[1:], g, side="right"))
        starts[g] = offsets[fi] + (g - cum[fi]) + seq_len
    return starts


def score_clean(pred_concat, series_list, low=LOW, high=HIGH, seq_len=SEQ, pred_len=PRED,
                eval_steps=15, clean_steps=30):
    """Pooled clean-window recall/FAR/lead — mirrors utils/batch_metrics.py."""
    file_lengths = [len(s) for s in series_list]
    starts = build_window_starts(seq_len, pred_len, file_lengths)
    true_series = np.concatenate(series_list)
    pt = pred_concat[:, :eval_steps]
    true_fut = np.stack([true_series[st:st + eval_steps] for st in starts])
    pa = (pt > high) | (pt < low)
    ta = (true_fut > high) | (true_fut < low)
    half = eval_steps // 2
    pred_last = pa[:, half:].any(1)
    true_last = ta[:, half:].any(1)
    true_patch = ta.any(1)
    alarm_series = (true_series > high) | (true_series < low)
    input_clean = np.array([not alarm_series[max(0, st - clean_steps):st].any() for st in starts])
    tlc = true_last & input_clean
    tpc = true_patch & input_clean
    recall = (pred_last & tlc).sum() / max(tlc.sum(), 1)
    negm = (~tpc) & input_clean
    far = (pred_last & negm).sum() / max(negm.sum(), 1)
    pred_patch = pa.any(1)
    leads = []
    for (s, _) in contiguous_events(alarm_series):
        cand = np.where(pred_patch & input_clean & (starts <= s) & (s <= starts + eval_steps - 1))[0]
        leads.append(0 if cand.size == 0 else max(0, min(s - int(starts[cand].min()), eval_steps)))
    lead = float(np.mean(leads)) if leads else 0.0
    return float(recall), float(far), lead, int(tlc.sum())


def load_series(split_path):
    cfg = io.load_yaml(split_path)
    return [io.read_target(DATA, rel, TGT) for rel in cfg["test"]]


def load_stack(dirs):
    preds = [io.target_view(np.load(Path(d) / "pred.npy"))[:, :H] for d in dirs]
    return np.stack(preds, axis=0)  # (N, Btot, H)


def train_gate(x, stack_t, true_t, y, clean, lead, cfg, epochs=800, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    gate = GateMLPMulti(8, 32, H, stack_t.shape[0], cfg["tau_soft"])
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        w = gate(x)
        fused = torch.einsum("bhn,nbh->bh", w, stack_t)
        loss, _ = alarm_aware_loss(fused, true_t, y, clean, lead, LOW, HIGH,
                                   tau_a=cfg["tau_a"], lambda_far=cfg["lambda_far"],
                                   lambda_lead=cfg["lambda_lead"], lambda_mse=0.1, half_start=H // 2)
        loss.backward(); opt.step()
    return gate


def validate():
    """Reproduce phase-0 gate Run9-10 clean metrics against the batch_metrics summary."""
    pred = io.target_view(np.load(ROOT / "results/ensemble_Gate_alarm_XMEAS10_phase0_test/pred.npy"))
    series = load_series(ROOT / "setting/TEP_IDV13_XMEAS10_mag25.yaml")
    r, f, l, npos = score_clean(pred, series)
    print(f"phase0 Run9-10 scorer: recall={r:.3f} FAR={f:.3f} lead={l*3:.2f}min n_pos={npos}")
    print("expected (batch_metrics): recall=0.491 FAR=0.024 lead=4.65min")
    ok = abs(r - 0.491) < 0.01 and abs(f - 0.024) < 0.01
    print("SCORER MATCH:", ok)
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        raise SystemExit(0 if validate() else 1)

    gate_split = ROOT / "setting/TEP_IDV13_XMEAS10_gate25.yaml"      # test=Run11-17
    series_list = load_series(gate_split)                            # 7 runs
    counts = [io.usable_count(len(s), SEQ, PRED) for s in series_list]
    run_id = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    exp_dirs = {"diff": "mag25_diff_gate25", "raw": "mag25_raw_gate25",
                "time_moe": "fm_time_moe_xmeas10_gate25", "sundial": "fm_sundial_xmeas10_gate25"}
    stack = load_stack([ROOT / "results" / exp_dirs[e] for e in EXPERTS])  # (N,Btot,H)
    true_t = io.target_view(np.load(ROOT / "results" / exp_dirs["diff"] / "true.npy"))[:, :H]
    feat = io.context_features(DATA, gate_split, TGT, SEQ, PRED, LOW, HIGH)
    y, clean, lead = _window_labels(DATA, gate_split, TGT, SEQ, PRED, H, LOW, HIGH)
    assert len(feat) == stack.shape[1] == len(run_id), (len(feat), stack.shape, len(run_id))

    grid = [dict(tau_soft=ts, lambda_far=lf, lambda_lead=ll)
            for ts in (0.3, 0.5, 1.0) for lf in (0.5, 1.0) for ll in (1.0, 4.0, 8.0)]
    grid = [dict(g, tau_a=0.003) for g in grid]

    rows = []
    for ci, cfg in enumerate(grid):
        oof = np.zeros((stack.shape[1], H), dtype=np.float64)
        for held in range(len(counts)):
            tr = run_id != held
            te = run_id == held
            mean = feat[tr].mean(0); scale = feat[tr].std(0) + 1e-6
            xtr = torch.tensor((feat[tr] - mean) / scale, dtype=torch.float32)
            xte = torch.tensor((feat[te] - mean) / scale, dtype=torch.float32)
            gate = train_gate(xtr, torch.tensor(stack[:, tr], dtype=torch.float32),
                              torch.tensor(true_t[tr], dtype=torch.float32),
                              torch.tensor(y[tr]), torch.tensor(clean[tr]),
                              torch.tensor(lead[tr], dtype=torch.float32), cfg)
            gate.eval()
            with torch.no_grad():
                w = gate(xte).numpy()
            oof[te] = np.sum(w * np.transpose(stack[:, te], (1, 2, 0)), axis=-1)
        r, f, l, npos = score_clean(oof, series_list)
        S = r + (1 - f / 0.05) + l / 15
        rows.append(dict(cfg, recall=round(r, 4), FAR=round(f, 4),
                         lead_min=round(l * 3, 3), S=round(S, 4)))
        print(f"[{ci+1}/{len(grid)}] tau={cfg['tau_soft']} lf={cfg['lambda_far']} "
              f"ll={cfg['lambda_lead']}: recall={r:.3f} FAR={f:.3f} lead={l*3:.2f}min S={S:.3f}")

    df = pd.DataFrame(rows).sort_values("S", ascending=False)
    out = ROOT / "results/TEP_IDV13_XMEAS10_Summary/alarmgate_cv.csv"
    df.to_csv(out, index=False)
    print("\n=== CV ranking (by combined S, leave-one-run-out over Run11-17) ===")
    print(df.to_string(index=False))
    print(f"\nsaved {out}")
    best = df.iloc[0]
    print(f"WINNER: tau_soft={best.tau_soft} lambda_far={best.lambda_far} lambda_lead={best.lambda_lead}")


if __name__ == "__main__":
    main()
