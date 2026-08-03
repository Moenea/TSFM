# scripts/adaptation/foundation_experts/fuse_residual_alarm.py
"""Residual multivariate-combiner alarm head. Same on-disk I/O contract as
fuse_gate_alarm.py, but the combiner is NOT a convex softmax gate: a small MLP
reads [expert forecasts + multivariate short context + distance-to-limit +
expert spread] and outputs a free residual delta added to the base expert
(first --expert = anchor, e.g. diff-P5). Trained with the same alarm_aware_loss
plus lambda_reg * ||delta||^2. The free residual can push the corrected forecast
OUTSIDE the expert convex hull, so it can correct a bias all experts share
(which a convex gate provably cannot). New file — fuse_gate_alarm.py untouched."""
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
from loss import alarm_aware_loss  # noqa: E402


class ResidualHead(nn.Module):
    """MLP mapping combined features -> per-step residual delta (B, H).
    Last layer is zero-initialised so delta starts at 0 => the corrected
    forecast starts exactly at the anchor expert, and training departs from it."""

    def __init__(self, in_dim, hidden, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)  # (B, H) residual delta


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
    """Read (low, high) from a limits CSV shaped like limits_tep_xmeas10.csv
    (high = iloc[1], low = iloc[2]) — same convention batch_metrics uses."""
    df = pd.read_csv(path, index_col=0)
    high = float(df.loc[target].iloc[1])
    low = float(df.loc[target].iloc[2])
    return low, high


def _window_labels(data_root, split_path, target, seq_len, pred_len, horizon, low, high,
                   dt=0.05, onset_h=30.0):
    """Per-window y_alarm (latter-half true crosses), clean (30-step context clear),
    lead_w — concatenated across the split's `test` files (same order as the stack)."""
    from loss import lead_weights
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


def _multivar_context(data_root, split_path, target, seq_len, pred_len, lookback):
    """Last `lookback` steps of ALL channels per window, flattened (B, n_ch*lookback).
    Iterates windows in the SAME order as io.context_features / the expert stack."""
    cfg = io.load_yaml(split_path)
    rows, n_ch = [], None
    for rel in cfg["test"]:
        arr = pd.read_csv(Path(data_root) / rel).to_numpy(dtype=np.float64)  # (rows, n_ch)
        if n_ch is None:
            n_ch = arr.shape[1]
        elif arr.shape[1] != n_ch:
            raise ValueError(f"channel-count mismatch in {rel}: {arr.shape[1]} vs {n_ch}")
        for local in range(io.usable_count(len(arr), seq_len, pred_len)):
            start = local + seq_len
            rows.append(arr[start - lookback:start, :].reshape(-1))  # (lookback*n_ch,)
    return np.asarray(rows, dtype=np.float64), int(n_ch)


def _build_feats(stack, mv_ctx, low, high):
    """stack: (N,B,H); mv_ctx: (B, n_ch*lookback). Returns (B, D) feature matrix:
    [expert forecasts | multivariate context | high-anchor | anchor-low | expert spread]."""
    n, b, h = stack.shape
    experts = np.transpose(stack, (1, 0, 2)).reshape(b, n * h)  # (B, N*H)
    anchor = stack[0]                                          # (B,H) first expert = anchor
    dist_high = high - anchor                                  # (B,H)
    dist_low = anchor - low                                    # (B,H)
    spread = stack.max(axis=0) - stack.min(axis=0)            # (B,H)
    return np.concatenate([experts, mv_ctx, dist_high, dist_low, spread], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", action="append", required=True, help="NAME:DIR (test); first = anchor")
    p.add_argument("--val-expert", action="append", required=True, help="NAME:DIR (head-train); same order")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-split", type=Path, required=True, help="limits fallback (pre-onset, needs Time)")
    p.add_argument("--val-split", type=Path, required=True, help="head-training split (its `test` files)")
    p.add_argument("--test-split", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limits-csv", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--lookback", type=int, default=10, help="multivariate context length")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau-soft", type=float, default=1.0, help="accepted for CLI parity; unused (no softmax)")
    p.add_argument("--tau-a", type=float, default=0.003)
    p.add_argument("--lambda-far", type=float, default=1.0)
    p.add_argument("--lambda-lead", type=float, default=1.0)
    p.add_argument("--lambda-mse", type=float, default=0.1)
    p.add_argument("--lambda-reg", type=float, default=0.1, help="||delta||^2 penalty (FAR / overfit guard)")
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

    val_mv, n_ch = _multivar_context(a.data_root, a.val_split, a.target, a.seq_len, a.pred_len, a.lookback)
    test_mv, n_ch_t = _multivar_context(a.data_root, a.test_split, a.target, a.seq_len, a.pred_len, a.lookback)
    assert n_ch == n_ch_t, (n_ch, n_ch_t)
    assert len(val_mv) == val_stack.shape[1], (len(val_mv), val_stack.shape)
    assert len(test_mv) == test_stack.shape[1], (len(test_mv), test_stack.shape)

    val_feat = _build_feats(val_stack, val_mv, low, high)
    test_feat = _build_feats(test_stack, test_mv, low, high)

    y_alarm, clean, lead_w = _window_labels(a.data_root, a.val_split, a.target,
                                            a.seq_len, a.pred_len, a.horizon, low, high)
    assert len(y_alarm) == val_stack.shape[1], (len(y_alarm), val_stack.shape)

    mean = val_feat.mean(axis=0)
    scale = val_feat.std(axis=0) + 1e-6
    val_x = torch.tensor((val_feat - mean) / scale, dtype=torch.float32)
    test_x = torch.tensor((test_feat - mean) / scale, dtype=torch.float32)
    anchor_val = torch.tensor(val_stack[0], dtype=torch.float32)   # (B,H) raw diff
    val_y = torch.tensor(val_true, dtype=torch.float32)
    y_t = torch.tensor(y_alarm)
    clean_t = torch.tensor(clean)
    lead_t = torch.tensor(lead_w, dtype=torch.float32)

    head = ResidualHead(val_feat.shape[1], a.hidden, a.horizon)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    logs = []
    for _ in range(a.epochs):
        opt.zero_grad()
        delta = head(val_x)                                   # (B,H)
        corrected = anchor_val + delta
        loss, comp = alarm_aware_loss(corrected, val_y, y_t, clean_t, lead_t, low, high,
                                      tau_a=a.tau_a, lambda_far=a.lambda_far,
                                      lambda_lead=a.lambda_lead, lambda_mse=a.lambda_mse,
                                      half_start=a.horizon // 2)
        reg = (delta ** 2).mean()
        loss = loss + a.lambda_reg * reg
        comp["reg"] = float(reg)
        comp["loss"] = float(loss)
        loss.backward()
        opt.step()
        logs.append(comp)

    head.eval()
    with torch.no_grad():
        delta_test = head(test_x).numpy()
    corrected_test = test_stack[0] + delta_test               # (B,H)
    base_full = io.target_view(np.load(test_experts[0][1] / "pred.npy"))
    out_full = base_full.copy()
    out_full[:, :a.horizon] = corrected_test
    io.save_result(a.output_dir, out_full,
                   io.target_view(np.load(test_experts[0][1] / "true.npy")),
                   {"method": "Residual-alarm", "anchor": test_experts[0][0],
                    "experts": [nm for nm, _ in test_experts], "n_experts": n})
    np.save(a.output_dir / "delta.npy", delta_test.astype(np.float32))
    torch.save(head.state_dict(), a.output_dir / "head.pt")
    (a.output_dir / "fit_log.json").write_text(json.dumps({
        "method": "Residual-alarm",
        "anchor": test_experts[0][0],
        "experts": [nm for nm, _ in test_experts],
        "in_dim": int(val_feat.shape[1]), "n_channels": n_ch,
        "final_components": logs[-1],
        "hparams": {"lookback": a.lookback, "hidden": a.hidden, "tau_a": a.tau_a,
                    "lambda_far": a.lambda_far, "lambda_lead": a.lambda_lead,
                    "lambda_mse": a.lambda_mse, "lambda_reg": a.lambda_reg, "epochs": a.epochs},
        "delta_stats": {"mean": float(delta_test.mean()), "std": float(delta_test.std()),
                        "frac_positive": float((delta_test > 0).mean())},
        "control_limits": {"low": low, "high": high, "source": str(a.limits_csv or a.train_split)},
    }, indent=2), encoding="utf-8")
    print(json.dumps({"final": logs[-1],
                      "delta_mean": float(delta_test.mean()),
                      "delta_frac_pos": float((delta_test > 0).mean()),
                      "limits": [low, high]}, indent=2))


if __name__ == "__main__":
    main()
