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


def base_true_full(base_dir: Path) -> np.ndarray:
    """Full-length true from the base expert, for save_result's true.npy."""
    return io.target_view(np.load(base_dir / "true.npy"))


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
    for k in range(1, n):  # alignment guard (round-trip float + diff-restore noise ~1e-4)
        if not np.allclose(val_trues[k], val_true, atol=1e-2):
            raise ValueError(f"val true mismatch for expert {val_experts[k][0]}")
        if not np.allclose(test_trues[k], test_true, atol=1e-2):
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

    io.save_result(args.output_dir, output_full, base_true_full(test_experts[0][1]),
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


if __name__ == "__main__":
    main()
