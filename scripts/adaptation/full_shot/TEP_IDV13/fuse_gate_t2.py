"""Train a context-only Gate-T2 on run8 and apply it to runs9-10."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


class GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def target_view(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        if array.shape[-1] != 1:
            raise ValueError(f"expected target-only predictions, got {array.shape}")
        return array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D/3D predictions, got {array.shape}")
    return array


def control_limits(data_root: Path, split_path: Path, target: str) -> tuple[float, float]:
    cfg = load_yaml(split_path)
    healthy = []
    for relative in cfg["train"]:
        frame = pd.read_csv(data_root / relative)
        healthy.append(frame.loc[frame["Time"] < 30.0, target].to_numpy(dtype=np.float64))
    values = np.concatenate(healthy)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return mean - 3.0 * std, mean + 3.0 * std


def context_features(
    data_root: Path,
    split_path: Path,
    target: str,
    seq_len: int,
    pred_len: int,
    low: float,
    high: float,
) -> np.ndarray:
    cfg = load_yaml(split_path)
    rows = []
    x = np.arange(96, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float((x_centered**2).sum())
    for relative in cfg["test"]:
        values = pd.read_csv(data_root / relative)[target].to_numpy(dtype=np.float64)
        usable = len(values) - seq_len - pred_len + 1
        for local in range(usable):
            start = local + seq_len
            long_window = values[start - seq_len : start]
            short_window = values[start - 96 : start]
            short_centered = short_window - short_window.mean()
            slope = float((x_centered * short_centered).sum() / denominator)
            last = float(values[start - 1])
            rows.append(
                [
                    last,
                    last - float(values[start - 2]),
                    float(short_window.mean()),
                    float(short_window.std(ddof=0)),
                    slope,
                    float(long_window.max() - long_window.min()),
                    last - low,
                    high - last,
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def load_result(path: Path, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_full = target_view(np.load(path / "pred.npy"))
    true_full = target_view(np.load(path / "true.npy"))
    return pred_full[:, :horizon], true_full[:, :horizon], pred_full


def mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((prediction - truth) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-train-split", type=Path, required=True)
    parser.add_argument("--raw-val-split", type=Path, required=True)
    parser.add_argument("--raw-test-split", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--val-diff-dir", type=Path, required=True)
    parser.add_argument("--val-raw-dir", type=Path, required=True)
    parser.add_argument("--test-diff-dir", type=Path, required=True)
    parser.add_argument("--test-raw-dir", type=Path, required=True)
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
    low, high = control_limits(args.data_root, args.raw_train_split, args.target)

    val_a, val_true, _ = load_result(args.val_diff_dir, args.horizon)
    val_b, val_true_b, _ = load_result(args.val_raw_dir, args.horizon)
    test_a, test_true, test_a_full = load_result(args.test_diff_dir, args.horizon)
    test_b, test_true_b, _ = load_result(args.test_raw_dir, args.horizon)
    if not np.allclose(val_true, val_true_b, atol=1e-4):
        raise ValueError("validation ground truth differs between experts")
    if not np.allclose(test_true, test_true_b, atol=1e-4):
        raise ValueError("test ground truth differs between experts")

    val_features = context_features(
        args.data_root, args.raw_val_split, args.target,
        args.seq_len, args.pred_len, low, high,
    )
    test_features = context_features(
        args.data_root, args.raw_test_split, args.target,
        args.seq_len, args.pred_len, low, high,
    )
    if len(val_features) != len(val_a) or len(test_features) != len(test_a):
        raise ValueError(
            f"feature/prediction mismatch: val {val_features.shape}/{val_a.shape}, "
            f"test {test_features.shape}/{test_a.shape}"
        )

    mean = val_features.mean(axis=0)
    scale = val_features.std(axis=0) + 1e-6
    val_x = torch.tensor((val_features - mean) / scale, dtype=torch.float32)
    test_x = torch.tensor((test_features - mean) / scale, dtype=torch.float32)
    val_a_t = torch.tensor(val_a, dtype=torch.float32)
    val_b_t = torch.tensor(val_b, dtype=torch.float32)
    val_y_t = torch.tensor(val_true, dtype=torch.float32)

    gate = GateMLP(8, args.hidden, args.horizon)
    optimizer = torch.optim.Adam(
        gate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    losses = []
    for _ in range(args.epochs):
        optimizer.zero_grad()
        weights = gate(val_x)
        prediction = weights * val_a_t + (1.0 - weights) * val_b_t
        loss = torch.mean((prediction - val_y_t) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    gate.eval()
    with torch.no_grad():
        test_weights = gate(test_x).numpy()
    fused = test_weights * test_a + (1.0 - test_weights) * test_b

    output_full = test_a_full.copy()
    output_full[:, : args.horizon] = fused
    original_shape = np.load(args.test_diff_dir / "pred.npy").shape
    if len(original_shape) == 3:
        output_to_save = output_full[:, :, None]
        true_to_save = target_view(np.load(args.test_diff_dir / "true.npy"))[:, :, None]
    else:
        output_to_save = output_full
        true_to_save = target_view(np.load(args.test_diff_dir / "true.npy"))

    per_window_a = np.mean((test_a - test_true) ** 2, axis=1)
    per_window_b = np.mean((test_b - test_true) ** 2, axis=1)
    oracle_window = np.where(
        (per_window_a <= per_window_b)[:, None], test_a, test_b
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "pred.npy", output_to_save.astype(np.float32))
    np.save(args.output_dir / "true.npy", true_to_save.astype(np.float32))
    np.save(args.output_dir / "weights.npy", test_weights.astype(np.float32))
    torch.save(gate.state_dict(), args.output_dir / "gate_t2.pt")
    log = {
        "method": "Gate-T2",
        "experts": ["Timer-XL-S-DIFF", "Timer-XL-S-raw"],
        "gate_train_split": "Run8",
        "test_split": ["Run9", "Run10"],
        "target": args.target,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "control_limits_train_only": {"low": low, "high": high},
        "final_gate_train_mse": losses[-1],
        "test_mse": {
            "diff": mse(test_a, test_true),
            "raw": mse(test_b, test_true),
            "gate_t2": mse(fused, test_true),
            "oracle_window": mse(oracle_window, test_true),
            "oracle_step": mse(
                np.where((test_a - test_true) ** 2 <= (test_b - test_true) ** 2, test_a, test_b),
                test_true,
            ),
        },
        "diff_expert_weight_mean_by_horizon": test_weights.mean(axis=0).tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }
    (args.output_dir / "fit_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
