"""Alarm-aware gate with expert-prediction features.

This is a feature-augmented companion to ``fuse_gate_alarm.py``.  In addition
to the eight context features, the gate sees each expert's forecast, distance
to both control limits, and inter-expert disagreement.  The training objective
is unchanged: alarm recall/FAR/lead dominate, with a small MSE regularizer.
"""
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
from fuse_gate_alarm import _limits_from_csv, _parse, _window_labels  # noqa: E402
from loss import alarm_aware_loss  # noqa: E402


class GateMLPFeatures(nn.Module):
    def __init__(self, in_dim: int, hidden: int, horizon: int, n_experts: int,
                 tau_soft: float):
        super().__init__()
        self.horizon, self.n_experts, self.tau_soft = horizon, n_experts, tau_soft
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).view(-1, self.horizon, self.n_experts) / self.tau_soft
        return torch.softmax(logits, dim=-1)


def _load_stack(experts, horizon):
    preds, trues = [], []
    for _, d in experts:
        preds.append(io.target_view(np.load(d / "pred.npy"))[:, :horizon])
        trues.append(io.target_view(np.load(d / "true.npy"))[:, :horizon])
    return np.stack(preds, axis=0), trues


def _features(context, stack, low, high):
    """Build context + forecast/margin/disagreement features per window."""
    # stack: (N, B, H). Keep absolute forecasts and normalize margin features
    # by the alarm-band width so the representation is scale-stable.
    n, b, h = stack.shape
    band = max(high - low, 1e-6)
    p = np.transpose(stack, (1, 0, 2))                    # (B,N,H)
    margins = np.stack([(p - high) / band, (low - p) / band], axis=2)
    spread = (p.max(axis=1) - p.min(axis=1)) / band       # (B,H)
    return np.concatenate([context, p.reshape(b, -1), margins.reshape(b, -1), spread], axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert", action="append", required=True, help="NAME:DIR (test)")
    p.add_argument("--val-expert", action="append", required=True, help="NAME:DIR (gate train)")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-split", type=Path, required=True)
    p.add_argument("--val-split", type=Path, required=True)
    p.add_argument("--test-split", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limits-csv", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau-soft", type=float, default=0.3)
    p.add_argument("--tau-a", type=float, default=0.003)
    p.add_argument("--lambda-far", type=float, default=1.0)
    p.add_argument("--lambda-lead", type=float, default=1.0)
    p.add_argument("--lambda-mse", type=float, default=0.1)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    test_experts, val_experts = _parse(a.expert), _parse(a.val_expert)
    n = len(test_experts)
    low, high = _limits_from_csv(a.limits_csv, a.target)
    val_stack, val_trues = _load_stack(val_experts, a.horizon)
    test_stack, test_trues = _load_stack(test_experts, a.horizon)
    val_true, test_true = val_trues[0], test_trues[0]
    for k in range(1, n):
        if not np.allclose(val_trues[k], val_true, atol=1e-2):
            raise ValueError(f"val true mismatch for {val_experts[k][0]}")
        if not np.allclose(test_trues[k], test_true, atol=1e-2):
            raise ValueError(f"test true mismatch for {test_experts[k][0]}")

    vc = io.context_features(a.data_root, a.val_split, a.target, a.seq_len, a.pred_len, low, high)
    tc = io.context_features(a.data_root, a.test_split, a.target, a.seq_len, a.pred_len, low, high)
    val_feat = _features(vc, val_stack, low, high)
    test_feat = _features(tc, test_stack, low, high)
    if len(val_feat) != val_stack.shape[1] or len(test_feat) != test_stack.shape[1]:
        raise ValueError("feature/prediction window count mismatch")
    mean, scale = val_feat.mean(0), val_feat.std(0) + 1e-6
    val_x = torch.tensor((val_feat - mean) / scale, dtype=torch.float32)
    test_x = torch.tensor((test_feat - mean) / scale, dtype=torch.float32)
    val_stack_t = torch.tensor(val_stack, dtype=torch.float32)
    y_alarm, clean, lead_w = _window_labels(a.data_root, a.val_split, a.target,
                                             a.seq_len, a.pred_len, a.horizon,
                                             low, high)
    if len(y_alarm) != val_stack.shape[1]:
        raise ValueError("label/prediction window count mismatch")

    y_t = torch.tensor(val_true, dtype=torch.float32)
    gate = GateMLPFeatures(val_feat.shape[1], a.hidden, a.horizon, n, a.tau_soft)
    opt = torch.optim.Adam(gate.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    logs = []
    for _ in range(a.epochs):
        opt.zero_grad()
        w = gate(val_x)
        fused = torch.einsum("bhn,nbh->bh", w, val_stack_t)
        loss, comp = alarm_aware_loss(
            fused, y_t, torch.tensor(y_alarm), torch.tensor(clean),
            torch.tensor(lead_w, dtype=torch.float32), low, high,
            tau_a=a.tau_a, lambda_far=a.lambda_far,
            lambda_lead=a.lambda_lead, lambda_mse=a.lambda_mse,
            half_start=a.horizon // 2)
        loss.backward(); opt.step(); logs.append(comp)

    gate.eval()
    with torch.no_grad():
        test_w = gate(test_x).numpy()
    fused_test = np.sum(test_w * np.transpose(test_stack, (1, 2, 0)), axis=-1)
    base_full = io.target_view(np.load(test_experts[0][1] / "pred.npy"))
    out_full = base_full.copy(); out_full[:, :a.horizon] = fused_test
    io.save_result(a.output_dir, out_full,
                   io.target_view(np.load(test_experts[0][1] / "true.npy")),
                   {"method": "Gate-alarm-predfeat", "experts": [x[0] for x in test_experts],
                    "feature_dim": int(val_feat.shape[1])})
    np.save(a.output_dir / "weights.npy", test_w.astype(np.float32))
    np.save(a.output_dir / "feature_mean.npy", mean.astype(np.float32))
    np.save(a.output_dir / "feature_scale.npy", scale.astype(np.float32))
    torch.save(gate.state_dict(), a.output_dir / "gate.pt")
    (a.output_dir / "fit_log.json").write_text(json.dumps({
        "method": "Gate-alarm-predfeat", "experts": [x[0] for x in test_experts],
        "feature_dim": int(val_feat.shape[1]), "final_components": logs[-1],
        "hparams": vars(a),
        "mean_weight_by_expert": test_w.mean(axis=(0, 1)).tolist(),
        "control_limits": {"low": low, "high": high},
    }, default=str, indent=2), encoding="utf-8")
    print(json.dumps({"final": logs[-1], "mean_w": test_w.mean(axis=(0, 1)).tolist(),
                      "feature_dim": int(val_feat.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
