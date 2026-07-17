# scripts/adaptation/foundation_experts/fuse_gate_alarm.py
"""Alarm-aware combined-score gate. Same on-disk I/O contract as fuse_gate_multi.py,
but the gate is trained on a soft-alarm recall/FAR/lead objective (not MSE), on a
disjoint fresh gate-training set (--val-*). +-3sigma is a hard threshold; control
limits come from --limits-csv (preferred) or --train-split's pre-onset. New file —
the validated fuse_gate_multi.py is untouched."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "gate_alarm"))
from common import expert_io as io  # noqa: E402
from loss import alarm_aware_loss, lead_weights  # noqa: E402


class GateMLPMulti(nn.Module):
    def __init__(self, in_dim, hidden, horizon, n_experts, tau_soft):
        super().__init__()
        self.horizon, self.n_experts, self.tau_soft = horizon, n_experts, tau_soft
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * n_experts),
        )

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


def _limits_from_csv(path: Path, target: str) -> tuple[float, float]:
    """Read (low, high) = (L, H) from a limits CSV shaped like limits_tep_xmeas10.csv
    (columns HH,H,L,LL; high = iloc[1], low = iloc[2]) — same convention batch_metrics uses."""
    df = pd.read_csv(path, index_col=0)
    high = float(df.loc[target].iloc[1])
    low = float(df.loc[target].iloc[2])
    return low, high


def _window_labels(data_root, split_path, target, seq_len, pred_len, horizon, low, high,
                   dt=0.05, onset_h=30.0):
    """Per-window y_alarm (latter-half true crosses), clean (30-step context clear),
    lead_w — concatenated across the split's `test` files, matching io.context_features order."""
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
    p.add_argument("--expert", action="append", required=True, help="NAME:DIR (test); first = base")
    p.add_argument("--val-expert", action="append", required=True, help="NAME:DIR (gate-train); same order")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-split", type=Path, required=True, help="limits fallback (pre-onset, needs Time)")
    p.add_argument("--val-split", type=Path, required=True, help="gate-training split (its `test` files)")
    p.add_argument("--test-split", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limits-csv", type=Path, default=None,
                   help="if set, read (low,high) from here instead of control_limits(train-split)")
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
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    test_experts, val_experts = _parse(a.expert), _parse(a.val_expert)
    n = len(test_experts)
    if a.limits_csv is not None:
        low, high = _limits_from_csv(a.limits_csv, a.target)
    else:
        low, high = io.control_limits(a.data_root, a.train_split, a.target)

    val_stack, val_trues = _load_stack(val_experts, a.horizon)
    test_stack, test_trues = _load_stack(test_experts, a.horizon)
    val_true = val_trues[0]
    for k in range(1, n):  # alignment guard (round-trip float + diff-restore noise ~1e-4)
        if not np.allclose(val_trues[k], val_true, atol=1e-2):
            raise ValueError(f"val true mismatch for expert {val_experts[k][0]}")
        if not np.allclose(test_trues[k], test_trues[0], atol=1e-2):
            raise ValueError(f"test true mismatch for expert {test_experts[k][0]}")

    val_feat = io.context_features(a.data_root, a.val_split, a.target, a.seq_len, a.pred_len, low, high)
    test_feat = io.context_features(a.data_root, a.test_split, a.target, a.seq_len, a.pred_len, low, high)
    assert len(val_feat) == val_stack.shape[1], (len(val_feat), val_stack.shape)
    assert len(test_feat) == test_stack.shape[1], (len(test_feat), test_stack.shape)

    y_alarm, clean, lead_w = _window_labels(a.data_root, a.val_split, a.target,
                                            a.seq_len, a.pred_len, a.horizon, low, high)
    assert len(y_alarm) == val_stack.shape[1], (len(y_alarm), val_stack.shape)

    mean = val_feat.mean(axis=0)
    scale = val_feat.std(axis=0) + 1e-6
    val_x = torch.tensor((val_feat - mean) / scale, dtype=torch.float32)
    test_x = torch.tensor((test_feat - mean) / scale, dtype=torch.float32)
    val_stack_t = torch.tensor(val_stack, dtype=torch.float32)
    val_y = torch.tensor(val_true, dtype=torch.float32)
    y_t = torch.tensor(y_alarm)
    clean_t = torch.tensor(clean)
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
        loss.backward()
        opt.step()
        logs.append(comp)

    gate.eval()
    with torch.no_grad():
        test_w = gate(test_x).numpy()
    fused_test = fuse(test_w, test_stack)
    base_full = io.target_view(np.load(test_experts[0][1] / "pred.npy"))
    out_full = base_full.copy()
    out_full[:, :a.horizon] = fused_test
    io.save_result(a.output_dir, out_full,
                   io.target_view(np.load(test_experts[0][1] / "true.npy")),
                   {"method": "Gate-alarm", "experts": [nm for nm, _ in test_experts], "n_experts": n})
    np.save(a.output_dir / "weights.npy", test_w.astype(np.float32))
    torch.save(gate.state_dict(), a.output_dir / "gate.pt")
    (a.output_dir / "fit_log.json").write_text(json.dumps({
        "method": "Gate-alarm",
        "experts": [nm for nm, _ in test_experts],
        "final_components": logs[-1],
        "hparams": {"tau_soft": a.tau_soft, "tau_a": a.tau_a, "lambda_far": a.lambda_far,
                    "lambda_lead": a.lambda_lead, "lambda_mse": a.lambda_mse, "epochs": a.epochs},
        "mean_weight_by_expert": test_w.mean(axis=(0, 1)).tolist(),
        "control_limits": {"low": low, "high": high, "source": str(a.limits_csv or a.train_split)},
    }, indent=2), encoding="utf-8")
    print(json.dumps({"final": logs[-1], "mean_w": test_w.mean(axis=(0, 1)).tolist(),
                      "limits": [low, high]}, indent=2))


if __name__ == "__main__":
    main()
