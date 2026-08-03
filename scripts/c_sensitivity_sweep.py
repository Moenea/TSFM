#!/usr/bin/env python3
"""Sensitivity of the early-warning metrics to the clean-window length C.

Backs the choice of C = 10 (30 min) used in the paper.  Everything else follows
``scripts/reproduce_paper_metrics.py`` exactly; only ``input_clean_steps`` is
varied.  C is the number of pre-origin samples that must all lie inside the
alarm band for a window to count as a genuine prognostic opportunity (Eq. 20).

Summary of the outcome (XMEAS10 / IDV13 / magnitude 25 / H = 5):

    C   min   pos   neg   onsets | Union@1%  keep   U@1%-base@100%  NaN
    10   30   101  1375      28  |   0.723   91%        +0.188        0
    15   45    76  1304      18  |   0.711   92%        +0.158        3
    20   60    54  1260      13  |   0.685   86%        +0.111        3
    25   75    42  1228      11  |   0.690   83%        +0.071        3
    30   90    40  1198       9  |   0.700   82%        +0.075        3

C = 10 keeps the largest sample (101 positives, 28 reachable onsets), the
highest 1%-to-100% retention, the largest margin of the 1% operators over the
best fully trained baseline, and is the only setting for which every cell of the
lead-time table is defined.  Its physical reading is also the simplest: the
process must have been healthy for 30 min, i.e. twice the H*dt = 15 min horizon.

Usage
-----
    python scripts/c_sensitivity_sweep.py
    python scripts/c_sensitivity_sweep.py --clean-steps 10 15 20 25 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.batch_metrics import build_window_starts, contiguous_events  # noqa: E402
from reproduce_paper_metrics import (  # noqa: E402
    ORDER, PCT, RATIOS, STEP_MIN, load_ground_truth,
)

BASELINES = ["STAConvBiLSTM", "TCNTransformer", "DiPCALSTM", "CNNLSTM", "LSTMGRU"]


def sweep(cfg, clean_steps_list):
    p = cfg["params"]
    eval_steps = int(p["eval_steps"])
    seq_len, pred_len = int(p["seq_len"]), int(p["pred_len"])
    results_root = ROOT / p.get("results_root", "results")

    _, alarm_series, file_lengths, high, low = load_ground_truth(cfg)
    events = contiguous_events(alarm_series)
    window_starts, _ = build_window_starts(seq_len, pred_len, file_lengths)
    window_starts = np.asarray(window_starts)
    clean = {C: np.array([not np.any(alarm_series[max(0, ws - C):ws])
                          for ws in window_starts], dtype=bool)
             for C in clean_steps_list}

    rows = []
    for entry in cfg["model_dirs"]:
        name = entry["name"]
        rdir = results_root / entry["result_dir"]
        if not (rdir / "pred.npy").exists():
            print(f"MISSING {name}", file=sys.stderr)
            continue
        pred = np.load(rdir / "pred.npy")
        true = np.load(rdir / "true.npy")
        pred = pred[:, :, -1] if pred.ndim == 3 else pred
        true = true[:, :, -1] if true.ndim == 3 else true
        pred, true = pred[:, :eval_steps], true[:, :eval_steps]
        H = pred.shape[1]
        pred_alarm = ((pred > high) | (pred < low)).any(axis=1)
        true_alarm = ((true > high) | (true < low)).any(axis=1)
        mdl, rt = name.split("||")
        for C in clean_steps_list:
            cl = clean[C]
            leads = []
            for start, _ in events:
                cand = np.where(pred_alarm & cl & (window_starts <= start)
                                & (start <= window_starts + H - 1))[0]
                if cand.size:
                    leads.append(max(1, min(start - int(window_starts[cand].min()) + 1, H)))
            rows.append(dict(mdl=mdl, rt=rt, C=C,
                             recall=pred_alarm[true_alarm & cl].mean(),
                             FAR=pred_alarm[(~true_alarm) & cl].mean(),
                             lead_min=np.mean(leads) * STEP_MIN if leads else np.nan,
                             n_events_anticipated=len(leads)))
    return pd.DataFrame(rows), events, window_starts, true_alarm, clean, eval_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "setting/bm_fullsweep.yaml"))
    ap.add_argument("--clean-steps", type=int, nargs="+", default=[10, 15, 20, 25, 30])
    ap.add_argument("--out", default=str(ROOT / "results/c_sensitivity_sweep.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    df, events, ws, true_alarm, clean, H = sweep(cfg, args.clean_steps)
    df.to_csv(args.out, index=False)

    print(f"{'C':>3} {'min':>4} {'clean':>6} {'pos':>5} {'neg':>6} {'onsets':>7} "
          f"{'U@1%':>7} {'U@100%':>7} {'keep':>6} {'margin':>7} {'NaN':>4}")
    for C in args.clean_steps:
        cl = clean[C]
        reach = sum(1 for s, _ in events
                    if np.any(cl & (ws <= s) & (s <= ws + H - 1)))
        s = df[df.C == C].set_index(["mdl", "rt"])
        u1, u100 = s.loc[("union", "r0p01"), "recall"], s.loc[("union", "r1p0"), "recall"]
        best = max(s.loc[(m, "r1p0"), "recall"] for m in BASELINES)
        print(f"{C:>3} {C * STEP_MIN:>4.0f} {int(cl.sum()):>6} "
              f"{int((true_alarm & cl).sum()):>5} {int(((~true_alarm) & cl).sum()):>6} "
              f"{reach:>7} {u1:>7.3f} {u100:>7.3f} {u1 / u100:>6.1%} "
              f"{u1 - best:>+7.3f} {int(df[df.C == C].lead_min.isna().sum()):>4}")

    for metric, nd in [("recall", 3), ("FAR", 4), ("lead_min", 2)]:
        for C in args.clean_steps:
            piv = df[df.C == C].pivot_table(index="mdl", columns="rt", values=metric)
            piv = piv.reindex(index=[m for m in ORDER if m in piv.index], columns=RATIOS)
            piv.columns = [PCT[c] for c in piv.columns]
            print(f"\n=== {metric}  C={C} ===")
            print(piv.round(nd).to_string())
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
