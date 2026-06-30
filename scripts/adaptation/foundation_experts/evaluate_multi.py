# scripts/adaptation/foundation_experts/evaluate_multi.py
"""Per-expert + gate metrics for the heterogeneous-TSFM gate. Reuses the
prognosis metric functions from the validated few-shot evaluate.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT / "scripts/adaptation/few_shot/TEP_IDV13"))
from evaluate import limits, file_counts, forecast_metrics, prognosis_metrics, view  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", action="append", required=True, help="NAME:DIR")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()

    low, high = limits(args.data_root, args.split, args.target)
    counts = file_counts(args.data_root, args.split, args.seq_len, args.pred_len)
    report = {"target": args.target, "control_limits": {"low": low, "high": high},
              "horizon_steps": args.horizon, "models": {}}
    for item in args.expert:
        name, _, directory = item.partition(":")
        directory = Path(directory)
        pred = view(np.load(directory / "pred.npy"))
        true = view(np.load(directory / "true.npy"))
        report["models"][name] = {
            **forecast_metrics(pred, true, args.horizon),
            **prognosis_metrics(pred, true, counts, low, high, args.seq_len, args.horizon),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
