#!/usr/bin/env python3
"""Reproduce the early-warning metrics reported in the XMEAS10 / IDV13 mag-25 paper.

This is the authoritative reproducer for Tables 1-3 of the manuscript
(prognostic recall, false prognosis rate, mean lead time).  It deliberately does
NOT go through ``utils.batch_metrics.compute_metrics``: that function implements
an older, different convention and is left untouched so that every other
consumer of it keeps its exact behaviour.  The two differ in three ways:

  1. alarm rule.  The paper (Eqs. 19, 21, 22) flags a window when ANY of the
     H = eval_steps forecast samples leaves the band.  ``batch_metrics`` masks to
     the latter half of the horizon (``half_start = eval_steps // 2``).
  2. lead time.  The paper (Eq. 23) averages only over the set D of onsets the
     method actually anticipates, and counts the warning issued at the forecast
     origin, so the metric ranges over [dt, H*dt].  ``batch_metrics`` folds
     missed onsets in as zero lead and omits the origin, giving [0, H*dt].
  3. clean filter.  The paper uses C = 10 pre-origin samples (30 min);
     ``setting/bm_fullsweep.yaml`` is the single source of truth for C via
     ``params.input_clean_steps``.

Both helper functions (window construction, event segmentation) are imported
from ``utils.batch_metrics`` so the two paths cannot drift apart.

Usage
-----
    python scripts/reproduce_paper_metrics.py
    python scripts/reproduce_paper_metrics.py --config setting/bm_fullsweep.yaml
    python scripts/reproduce_paper_metrics.py --verify results/..._Summary_clean10.csv
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

STEP_MIN = 3.0  # TEP sampling interval
RATIOS = ["r0p01", "r0p05", "r0p1", "r0p25", "r0p5", "r1p0"]
PCT = dict(zip(RATIOS, ["1%", "5%", "10%", "25%", "50%", "100%"]))
ORDER = ["union", "diff2", "diff", "raw", "Time-MoE", "Sundial",
         "STAConvBiLSTM", "TCNTransformer", "DiPCALSTM", "CNNLSTM", "LSTMGRU"]


def load_ground_truth(cfg):
    """Concatenated target series of the test runs + its alarm indicator."""
    p = cfg["params"]
    target = p["target"]
    limits = pd.read_csv(p["limit_csv_path"], index_col=0)
    high = float(limits.loc[target].iloc[1])   # H limit
    low = float(limits.loc[target].iloc[2])    # L limit

    data_root = Path(p["data_root"])
    series, file_lengths = [], []
    for name in cfg["test"]:
        arr = pd.read_csv(data_root / name)[target].to_numpy(copy=True)
        series.append(arr)
        file_lengths.append(len(arr))
    series = np.concatenate(series, axis=0)
    return series, (series > high) | (series < low), file_lengths, high, low


def compute(cfg, verbose=True):
    p = cfg["params"]
    eval_steps = int(p["eval_steps"])
    clean_steps = int(p["input_clean_steps"])
    seq_len, pred_len = int(p["seq_len"]), int(p["pred_len"])
    results_root = ROOT / p.get("results_root", "results")

    _, alarm_series, file_lengths, high, low = load_ground_truth(cfg)
    events = contiguous_events(alarm_series)
    window_starts, _ = build_window_starts(seq_len, pred_len, file_lengths)
    window_starts = np.asarray(window_starts)

    # Eq. (20): the clean_steps samples before the forecast origin must be in band
    clean = np.array([not np.any(alarm_series[max(0, ws - clean_steps):ws])
                      for ws in window_starts], dtype=bool)

    rows, diag = [], None
    for entry in cfg["model_dirs"]:
        name = entry["name"]
        rdir = results_root / entry["result_dir"]
        if not (rdir / "pred.npy").exists() or not (rdir / "true.npy").exists():
            print(f"MISSING {name}  ({rdir})", file=sys.stderr)
            continue
        pred = np.load(rdir / "pred.npy")
        true = np.load(rdir / "true.npy")
        pred = pred[:, :, -1] if pred.ndim == 3 else pred
        true = true[:, :, -1] if true.ndim == 3 else true
        pred, true = pred[:, :eval_steps], true[:, :eval_steps]
        H = pred.shape[1]

        # Eq. (19): "any of the H steps leaves the band", on truth and forecast
        pred_alarm = ((pred > high) | (pred < low)).any(axis=1)
        true_patch = (true > high) | (true < low)
        true_alarm = true_patch.any(axis=1)

        pos = true_alarm & clean
        neg = (~true_alarm) & clean
        recall = float(pred_alarm[pos].mean()) if pos.any() else np.nan   # Eq. (21)
        far = float(pred_alarm[neg].mean()) if neg.any() else np.nan      # Eq. (22)

        # Eq. (23): mean lead over D = onsets this method anticipates
        leads = []
        for start, _ in events:
            cand = np.where(pred_alarm & clean
                            & (window_starts <= start)
                            & (start <= window_starts + H - 1))[0]
            if cand.size == 0:
                continue                      # onset not in D -> excluded
            earliest = int(window_starts[cand].min())
            leads.append(max(1, min(start - earliest + 1, H)))
        lead_min = float(np.mean(leads)) * STEP_MIN if leads else np.nan

        if diag is None:
            diag = (int(pos.sum()), int(neg.sum()), int(true_patch[pos][:, 0].sum()))
        mdl, rt = name.split("||")
        rows.append(dict(mdl=mdl, rt=rt, recall=recall, FAR=far,
                         lead_min=lead_min, n_events_anticipated=len(leads)))

    df = pd.DataFrame(rows)
    if verbose and diag is not None:
        n_pos, n_neg, n_cross0 = diag
        n_reach = sum(1 for s, _ in events
                      if np.any(clean & (window_starts <= s)
                                & (s <= window_starts + eval_steps - 1)))
        print(f"target={p['target']}  band=({low:.6f}, {high:.6f})  "
              f"H={eval_steps}  C={clean_steps} ({clean_steps * STEP_MIN:.0f} min)")
        print(f"windows={len(window_starts)}  clean={int(clean.sum())}  "
              f"positives={n_pos}  negatives={n_neg}")
        print(f"onset events={len(events)}  reachable under the clean filter={n_reach}")
        print(f"of the positives, already out of band at step 1: {n_cross0} "
              f"({100 * n_cross0 / max(n_pos, 1):.1f}%)")
    return df


def show(df):
    for metric, nd in [("recall", 3), ("FAR", 4), ("lead_min", 2)]:
        piv = df.pivot_table(index="mdl", columns="rt", values=metric)
        piv = piv.reindex(index=[m for m in ORDER if m in piv.index], columns=RATIOS)
        piv.columns = [PCT[c] for c in piv.columns]
        print(f"\n=== {metric} ===")
        print(piv.round(nd).to_string())


def verify(df, path, tol=1e-9):
    """Check recall/FAR against a previously frozen summary CSV."""
    ref = pd.read_csv(path)
    m = df.merge(ref, on=["mdl", "rt"], suffixes=("", "_ref"))
    if len(m) != len(ref):
        print(f"VERIFY: row-count mismatch {len(m)} vs {len(ref)}")
        return False
    ok = True
    for col in ("recall", "FAR"):
        d = (m[col] - m[f"{col}_ref"]).abs().max()
        print(f"VERIFY {col}: max |diff| = {d:.3e} over {len(m)} rows"
              f"  -> {'OK' if d <= tol else 'MISMATCH'}")
        ok &= d <= tol
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "setting/bm_fullsweep.yaml"))
    ap.add_argument("--out", default=None,
                    help="output CSV (default: results/<target>_Summary_paper.csv)")
    ap.add_argument("--verify", default=None,
                    help="frozen summary CSV to check recall/FAR against")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = compute(cfg)
    show(df)

    out = Path(args.out) if args.out else \
        ROOT / f"results/{cfg['params']['target']}_Summary_paper.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")

    if args.verify:
        sys.exit(0 if verify(df, args.verify) else 1)


if __name__ == "__main__":
    main()
