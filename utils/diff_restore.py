"""Helpers for converting differenced forecasts back to raw target values."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml


def _load_test_files(root_path: str | Path, raw_split_file: str | Path) -> list[Path]:
    root = Path(root_path)
    split_path = Path(raw_split_file)
    with split_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    files = cfg.get('test', []) or []
    if not files:
        raise ValueError(f"raw split file has no test entries: {split_path}")
    return [root / p for p in files]


def _target_values(path: Path, target: str | None) -> np.ndarray:
    df = pd.read_csv(path)
    if len(df) > 0 and isinstance(df[df.columns[0]].iloc[0], str):
        df = df[df.columns[1:]]
    if target is None:
        target = str(df.columns[-1])
    if target not in df.columns:
        raise ValueError(f"target {target!r} not found in {path}; columns={list(df.columns)}")
    return df[target].to_numpy(dtype=np.float64, copy=True)


def _window_bases(root_path: str | Path, raw_split_file: str | Path,
                  target: str | None, seq_len: int, horizon: int) -> np.ndarray:
    bases = []
    total = 0
    for path in _load_test_files(root_path, raw_split_file):
        values = _target_values(path, target)
        usable = max(0, len(values) - int(seq_len) - int(horizon) + 1)
        if usable <= 0:
            continue
        idx = np.arange(usable, dtype=np.int64) + int(seq_len) - 1
        bases.append(values[idx])
        total += usable
    if not bases:
        raise ValueError(
            f"no usable raw windows for seq_len={seq_len}, horizon={horizon}, split={raw_split_file}")
    out = np.concatenate(bases, axis=0)
    if out.shape[0] != total:
        raise AssertionError("internal window base count mismatch")
    return out


def _restore_one(diff: np.ndarray, bases: np.ndarray) -> np.ndarray:
    if diff.ndim == 2:
        if diff.shape[0] != bases.shape[0]:
            raise ValueError(
                f"row count mismatch: diff has {diff.shape[0]} rows, raw split expects {bases.shape[0]}. "
                "Regenerate 2103ts_diff with aligned first differences if needed.")
        return bases[:, None] + np.cumsum(diff, axis=1)
    if diff.ndim == 3:
        if diff.shape[0] != bases.shape[0]:
            raise ValueError(
                f"row count mismatch: diff has {diff.shape[0]} rows, raw split expects {bases.shape[0]}. "
                "Regenerate 2103ts_diff with aligned first differences if needed.")
        if diff.shape[2] != 1:
            raise ValueError(
                f"restore_diff_to_raw currently expects target-only outputs for 3D arrays; got shape {diff.shape}")
        restored = bases[:, None] + np.cumsum(diff[:, :, 0], axis=1)
        return restored[:, :, None]
    raise ValueError(f"expected 2D or 3D diff array, got shape {diff.shape}")


def restore_diff_to_raw(pred_diff: np.ndarray, true_diff: np.ndarray, *,
                        root_path: str | Path, raw_split_file: str | Path,
                        target: str | None, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate predicted/true first differences onto raw PI10102 bases.

    The differenced CSVs are expected to be raw-aligned and equal length to the
    raw files, with d[0]=0 and d[t]=raw[t]-raw[t-1]. For each test window, TSFM
    predicts d[start:start+horizon], so the raw base is raw[start-1].
    """
    if pred_diff.shape[0] != true_diff.shape[0] or pred_diff.shape[1] != true_diff.shape[1]:
        raise ValueError(f"pred/true shape mismatch: {pred_diff.shape} vs {true_diff.shape}")
    horizon = int(pred_diff.shape[1])
    bases = _window_bases(root_path, raw_split_file, target, int(seq_len), horizon)
    return _restore_one(pred_diff, bases), _restore_one(true_diff, bases)
