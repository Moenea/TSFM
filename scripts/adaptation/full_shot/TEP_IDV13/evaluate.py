"""Evaluate TEP IDV13 forecasts and label-aware prognosis on runs9-10."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def view(array: np.ndarray) -> np.ndarray:
    return array[:, :, 0] if array.ndim == 3 else array


def limits(data_root: Path, split: Path, target: str) -> tuple[float, float]:
    cfg = load_yaml(split)
    chunks = []
    for relative in cfg["train"]:
        frame = pd.read_csv(data_root / relative)
        chunks.append(frame.loc[frame["Time"] < 30.0, target].to_numpy(dtype=float))
    values = np.concatenate(chunks)
    return float(values.mean() - 3 * values.std()), float(values.mean() + 3 * values.std())


def file_counts(data_root: Path, split: Path, seq_len: int, pred_len: int) -> list[int]:
    return [
        len(pd.read_csv(data_root / relative)) - seq_len - pred_len + 1
        for relative in load_yaml(split)["test"]
    ]


def forecast_metrics(pred: np.ndarray, true: np.ndarray, horizon: int) -> dict:
    error = pred[:, :horizon] - true[:, :horizon]
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def prognosis_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    counts: list[int],
    low: float,
    high: float,
    seq_len: int,
    horizon: int,
    dt: float = 0.05,
    onset_h: float = 30.0,
) -> dict:
    offset = 0
    event_hits = []
    lead_times = []
    false_before = 0
    opportunities_before = 0
    window_tp = window_fp = window_tn = window_fn = 0
    for count in counts:
        p = pred[offset : offset + count, :horizon]
        y = true[offset : offset + count, :horizon]
        offset += count
        pred_alarm = ((p < low) | (p > high)).any(axis=1)
        true_alarm = ((y < low) | (y > high)).any(axis=1)
        window_tp += int((pred_alarm & true_alarm).sum())
        window_fp += int((pred_alarm & ~true_alarm).sum())
        window_tn += int((~pred_alarm & ~true_alarm).sum())
        window_fn += int((~pred_alarm & true_alarm).sum())

        starts = np.arange(count) + seq_len
        issue_times = starts * dt
        fully_pre_onset = (starts + horizon - 1) * dt < onset_h
        false_before += int((pred_alarm & fully_pre_onset).sum())
        opportunities_before += int(fully_pre_onset.sum())

        true_positions = np.argwhere((y < low) | (y > high))
        true_crossing_times = np.asarray(
            [(starts[row] + step) * dt for row, step in true_positions],
            dtype=float,
        )
        true_crossing_times = true_crossing_times[true_crossing_times >= onset_h]
        if len(true_crossing_times) == 0:
            event_hits.append(False)
            continue
        actual_alarm_h = float(true_crossing_times.min())
        candidates = []
        for row in range(count):
            if issue_times[row] >= actual_alarm_h:
                continue
            alarm_steps = np.flatnonzero((p[row] < low) | (p[row] > high))
            if len(alarm_steps) == 0:
                continue
            predicted_crossing_h = float((starts[row] + alarm_steps[0]) * dt)
            if predicted_crossing_h >= onset_h:
                candidates.append(float(issue_times[row]))
        hit = bool(candidates)
        event_hits.append(hit)
        if hit:
            lead_times.append(actual_alarm_h - min(candidates))

    return {
        "event_recall": float(np.mean(event_hits)),
        "mean_lead_time_h": float(np.mean(lead_times)) if lead_times else None,
        "pre_onset_false_alarm_rate": false_before / max(opportunities_before, 1),
        "window_precision": window_tp / max(window_tp + window_fp, 1),
        "window_recall": window_tp / max(window_tp + window_fn, 1),
        "window_counts": {"tp": window_tp, "fp": window_fp, "tn": window_tn, "fn": window_fn},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--diff-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()

    low, high = limits(args.data_root, args.split, args.target)
    counts = file_counts(args.data_root, args.split, args.seq_len, args.pred_len)
    models = {"raw": args.raw_dir, "diff": args.diff_dir, "gate_t2": args.gate_dir}
    report = {
        "target": args.target,
        "control_limits": {"low": low, "high": high},
        "test_runs": [9, 10],
        "horizon_steps": args.horizon,
        "sample_interval_h": 0.05,
        "models": {},
    }
    for name, directory in models.items():
        pred = view(np.load(directory / "pred.npy"))
        true = view(np.load(directory / "true.npy"))
        report["models"][name] = {
            **forecast_metrics(pred, true, args.horizon),
            **prognosis_metrics(
                pred, true, counts, low, high, args.seq_len, args.horizon
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
