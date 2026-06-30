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
