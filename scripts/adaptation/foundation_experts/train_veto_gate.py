"""Train a Union-veto gate and apply it to one test expert pair.

Union remains the high-recall candidate.  A scalar gate accepts a candidate
window only when its learned probability exceeds a threshold selected on the
gate-validation runs subject to FAR <= 3%.  Rejected windows fall back to the
diff expert, which supplies the low-FAR safe forecast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "foundation_experts"))
sys.path.insert(0, str(HERE))
from common import expert_io as io  # noqa: E402
from fuse_gate_alarm import _limits_from_csv, _parse, _window_labels  # noqa: E402


class VetoMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load(experts, horizon):
    ps, ts = [], []
    for _, d in experts:
        ps.append(io.target_view(np.load(d / "pred.npy"))[:, :horizon])
        ts.append(io.target_view(np.load(d / "true.npy"))[:, :horizon])
    return np.stack(ps, 0), ts


def _features(context, stack, low, high):
    """Context + candidate/margin/agreement features, one row per window."""
    p = np.transpose(stack, (1, 0, 2))  # B,N,H
    band = max(high - low, 1e-6)
    half = p.shape[2] // 2
    hi = p > high; lo = p < low
    cand = (hi[:, :, half:] | lo[:, :, half:]).any(axis=(1, 2)).astype(float)
    both = ((hi[:, 0, half:] | lo[:, 0, half:]) &
            (hi[:, 1, half:] | lo[:, 1, half:])).any(axis=1).astype(float)
    n_cross = np.stack([(hi[:, i, half:] | lo[:, i, half:]).sum(axis=1)
                        for i in range(p.shape[1])], axis=1) / max(1, p.shape[2] - half)
    margin = np.stack([
        (p.max(axis=(1, 2)) - high) / band, (low - p.min(axis=(1, 2))) / band,
        (p[:, 0, half:].max(1) - high) / band,
        (low - p[:, 0, half:].min(1)) / band,
        (p[:, 1, half:].max(1) - high) / band,
        (low - p[:, 1, half:].min(1)) / band,
        (p[:, 0, half:].max(1) - p[:, 1, half:].max(1)) / band,
        (p[:, 0, half:].min(1) - p[:, 1, half:].min(1)) / band,
    ], axis=1)
    return np.concatenate([context, cand[:, None], both[:, None], n_cross, margin], axis=1)


def _alarm(pred, low, high, half):
    return ((pred[:, half:] > high) | (pred[:, half:] < low)).any(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert", action="append", required=True, help="diff:DIR then diff2:DIR (test)")
    p.add_argument("--val-expert", action="append", required=True, help="diff:DIR then diff2:DIR")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--val-split", type=Path, required=True)
    p.add_argument("--test-split", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--limits-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--far-budget", type=float, default=0.03)
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    low, high = _limits_from_csv(args.limits_csv, args.target)
    test_experts, val_experts = _parse(args.expert), _parse(args.val_expert)
    val_stack, val_trues = _load(val_experts, args.horizon)
    test_stack, test_trues = _load(test_experts, args.horizon)
    for k in range(1, len(val_experts)):
        if not np.allclose(val_trues[k], val_trues[0], atol=1e-2):
            raise ValueError("validation truth mismatch")
        if not np.allclose(test_trues[k], test_trues[0], atol=1e-2):
            raise ValueError("test truth mismatch")
    vc = io.context_features(args.data_root, args.val_split, args.target,
                             args.seq_len, args.pred_len, low, high)
    tc = io.context_features(args.data_root, args.test_split, args.target,
                             args.seq_len, args.pred_len, low, high)
    vx, tx = _features(vc, val_stack, low, high), _features(tc, test_stack, low, high)
    mean, scale = vx.mean(0), vx.std(0) + 1e-6
    vx = torch.tensor((vx - mean) / scale, dtype=torch.float32)
    tx = torch.tensor((tx - mean) / scale, dtype=torch.float32)
    y, clean, _ = _window_labels(args.data_root, args.val_split, args.target,
                                 args.seq_len, args.pred_len, args.horizon, low, high)
    y_t = torch.tensor(y.astype(np.float32))
    # Give positives a modest extra weight; FAR is enforced separately when
    # selecting the operating threshold.
    pos_weight = torch.tensor(max(1.0, float((~y).sum() / max(1, y.sum())) ** 0.5))
    gate = VetoMLP(vx.shape[1], args.hidden)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(args.epochs):
        opt.zero_grad(); loss = loss_fn(gate(vx), y_t); loss.backward(); opt.step()
    gate.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(gate(vx)).numpy()
        test_p = torch.sigmoid(gate(tx)).numpy()

    # Select the highest-recall threshold on validation subject to the FAR
    # budget. Use only clean windows for both rates, matching batch_metrics.
    val_hi, val_lo = val_stack.max(axis=0), val_stack.min(axis=0)
    val_union = val_stack[0].copy()
    val_union = np.where(val_hi > high, val_hi, val_union)
    val_union = np.where(val_lo < low, val_lo, val_union)
    safe = val_stack[0]
    thresholds = np.unique(np.r_[0.0, val_p, 1.0])
    best = None
    for q in thresholds:
        accept = val_p >= q
        out = np.where(accept[:, None], val_union, safe)
        pa = _alarm(out, low, high, args.horizon // 2)
        rec = float(pa[y & clean].mean()) if np.any(y & clean) else 0.0
        far = float(pa[(~y) & clean].mean()) if np.any((~y) & clean) else 0.0
        score = (rec, -far)
        if far <= args.far_budget and (best is None or score > best[0]):
            best = (score, float(q), rec, far)
    if best is None:
        # Conservative fallback: maximize recall among thresholds and report
        # that the requested budget was infeasible on validation.
        q = float(thresholds[np.argmin([_alarm(np.where((val_p >= z)[:, None], val_union, safe), low, high,
                                                   args.horizon // 2)[(~y) & clean].mean()
                                         for z in thresholds])])
        best = ((0.0, 0.0), q, 0.0, 1.0)
    threshold = best[1]
    test_hi, test_lo = test_stack.max(axis=0), test_stack.min(axis=0)
    test_union = test_stack[0].copy()
    test_union = np.where(test_hi > high, test_hi, test_union)
    test_union = np.where(test_lo < low, test_lo, test_union)
    test_out = np.where((test_p >= threshold)[:, None], test_union, test_stack[0])
    base = io.target_view(np.load(test_experts[0][1] / "pred.npy")).copy()
    base[:, :args.horizon] = test_out
    args.output_dir.mkdir(parents=True, exist_ok=True)
    io.save_result(args.output_dir, base,
                   io.target_view(np.load(test_experts[0][1] / "true.npy")),
                   {"method": "Union-veto-gate", "threshold": threshold,
                    "val_recall": best[2], "val_FAR": best[3]})
    np.save(args.output_dir / "gate_probability.npy", test_p.astype(np.float32))
    np.save(args.output_dir / "feature_mean.npy", mean.astype(np.float32))
    np.save(args.output_dir / "feature_scale.npy", scale.astype(np.float32))
    torch.save(gate.state_dict(), args.output_dir / "gate.pt")
    (args.output_dir / "fit_log.json").write_text(json.dumps({
        "method": "Union-veto-gate", "threshold": threshold,
        "val_recall": best[2], "val_FAR": best[3],
        "feature_dim": int(mean.size), "far_budget": args.far_budget,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "val_recall": best[2],
                      "val_FAR": best[3], "feature_dim": int(mean.size)}, indent=2))


if __name__ == "__main__":
    main()
