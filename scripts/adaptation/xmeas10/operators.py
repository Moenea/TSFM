"""Input-transform operators for TEP experts + their exact offline inverses.

Each operator maps a raw multivariate CSV to a transformed one (like
prepare_diff.py: same shape, aligned, transformed[0]=0). Timer-XL is fine-tuned
on the transformed data; at eval we restore the horizon-step forecast back to raw
units OFFLINE here, so exp_forecast.py is never touched (zero risk). The diff
inverse is golden-tested against the existing diff expert (which stores both
pred_diff.npy and the in-exp-restored pred.npy).

Operator families by restore op:
  integrate-on-restore (noise-suppressing): diff, diff2
  difference-on-restore (noise-amplifying):  cusum, ewma
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MU_PREONSET_ROWS = 500   # pre-onset baseline window (fault onset ~ row 600) for cusum mu
EWMA_ALPHA = 0.3


# ---------- forward transforms (columns = channels, axis 0 = time) ----------
def fwd_diff(x):
    d = np.zeros_like(x); d[1:] = np.diff(x, axis=0); return d


def fwd_diff2(x):
    dd = np.zeros_like(x); dd[2:] = np.diff(x, n=2, axis=0); return dd


def fwd_cusum(x):
    mu = x[:MU_PREONSET_ROWS].mean(axis=0)
    s = np.zeros_like(x); s[1:] = np.cumsum((x - mu)[1:], axis=0); return s


def fwd_ewma(x, alpha=EWMA_ALPHA):
    e = np.empty_like(x); e[0] = x[0]
    for t in range(1, len(x)):
        e[t] = alpha * x[t] + (1 - alpha) * e[t - 1]
    return e


def fwd_ma(x, w=5):
    """Causal trailing moving average per column (input feature only; no inverse)."""
    cs = np.cumsum(x, axis=0)
    out = np.empty_like(x)
    for t in range(len(x)):
        lo = t - w + 1
        out[t] = cs[t] / (t + 1) if lo <= 0 else (cs[t] - cs[lo - 1]) / w
    return out


FORWARD = {"diff": fwd_diff, "diff2": fwd_diff2, "cusum": fwd_cusum, "ewma": fwd_ewma}
MA_WINDOW = 5


def build_multiview(src_dir, dst_dir, target, glob):
    """Write 20-channel CSVs = raw | diff | diff2 | ma, target column = 'diff::<target>'
    placed LAST. Forecasting the diff-target keeps diff's edge; raw/diff2/ma are extra
    input views. All channels are causal (no future leakage)."""
    src, dst = Path(src_dir), Path(dst_dir); dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob(glob))
    if not paths:
        raise RuntimeError(f"no files matching {glob} in {src}")
    tgt_col = f"diff::{target}"
    for path in paths:
        df = pd.read_csv(path)
        cols = list(df.columns)              # csv_5var_lowmag has no Time column
        if target not in cols:
            raise ValueError(f"{target} not in {path}: {cols}")
        X = df[cols].to_numpy(dtype=np.float64)
        blocks = {"raw": X, "diff": fwd_diff(X), "diff2": fwd_diff2(X), "ma": fwd_ma(X, MA_WINDOW)}
        out = pd.DataFrame()
        for bn, B in blocks.items():
            for j, c in enumerate(cols):
                name = f"{bn}::{c}"
                if name != tgt_col:
                    out[name] = B[:, j]
        out[tgt_col] = blocks["diff"][:, cols.index(target)]   # target last
        out.to_csv(dst / path.name, index=False)
    print(f"[multiview] wrote {len(paths)} files ({out.shape[1]} ch, target={tgt_col!r}) -> {dst}")


# ---------- window bases from a series (raw or transformed) ----------
def _files(root, split_file):
    cfg = yaml.safe_load(Path(split_file).read_text(encoding="utf-8")) or {}
    return [Path(root) / p for p in (cfg.get("test") or [])]


def _target_series(path, target):
    df = pd.read_csv(path)
    if df.columns[0] == "Time":
        df = df[df.columns[1:]]
    if target is None:
        target = str(df.columns[-1])
    return df[target].to_numpy(dtype=np.float64)


def _per_window(root, split_file, target, seq_len, horizon, fn):
    """Concatenate fn(series, start) over every usable window of every test file."""
    out = []
    for path in _files(root, split_file):
        s = _target_series(path, target)
        usable = max(0, len(s) - seq_len - horizon + 1)
        for local in range(usable):
            out.append(fn(s, local + seq_len))
    return np.asarray(out, dtype=np.float64)


# ---------- inverses: transformed horizon forecast (B,H) -> raw (B,H) ----------
def inv_diff(p, root, raw_split, target, seq_len):
    b1 = _per_window(root, raw_split, target, seq_len, p.shape[1], lambda s, st: s[st - 1])
    return b1[:, None] + np.cumsum(p, axis=1)


def inv_diff2(p, root, raw_split, target, seq_len):
    h = p.shape[1]
    b1 = _per_window(root, raw_split, target, seq_len, h, lambda s, st: s[st - 1])
    d0 = _per_window(root, raw_split, target, seq_len, h, lambda s, st: s[st - 1] - s[st - 2])
    d = d0[:, None] + np.cumsum(p, axis=1)
    return b1[:, None] + np.cumsum(d, axis=1)


def inv_cusum(p, root, raw_split, target, seq_len, trans_root, trans_split):
    h = p.shape[1]
    mu = _per_window(root, raw_split, target, seq_len, h,
                     lambda s, st: s[:MU_PREONSET_ROWS].mean())
    tb = _per_window(trans_root, trans_split, target, seq_len, h, lambda s, st: s[st - 1])
    sfull = np.concatenate([tb[:, None], p], axis=1)          # (B,H+1)
    return mu[:, None] + np.diff(sfull, axis=1)


def inv_ewma(p, root, raw_split, target, seq_len, trans_root, trans_split, alpha=EWMA_ALPHA):
    tb = _per_window(trans_root, trans_split, target, seq_len, p.shape[1], lambda s, st: s[st - 1])
    raw = np.empty_like(p)
    prev = tb.copy()
    for k in range(p.shape[1]):
        raw[:, k] = (p[:, k] - (1 - alpha) * prev) / alpha
        prev = p[:, k]
    return raw


def _raw_windows(root, split, target, seq_len, horizon):
    """The raw target over [start:start+horizon] for every window (B,horizon)."""
    return _per_window(root, split, target, seq_len, horizon,
                       lambda s, st: s[st:st + horizon])


def restore_result(op, pred_dir, raw_root, raw_split, trans_root, trans_split,
                   target, seq_len, out_dir):
    """Offline-restore a transformed-space result dir (pred.npy/true.npy) to raw
    units, writing a new expert-contract dir. Sanity-checks restored ground truth
    against independently-read raw target windows."""
    pred = np.load(Path(pred_dir) / "pred.npy")
    p = pred[:, :, 0] if pred.ndim == 3 else pred
    h = p.shape[1]
    if op == "diff":
        rp = inv_diff(p, raw_root, raw_split, target, seq_len)
    elif op == "diff2":
        rp = inv_diff2(p, raw_root, raw_split, target, seq_len)
    elif op == "cusum":
        rp = inv_cusum(p, raw_root, raw_split, target, seq_len, trans_root, trans_split)
    elif op == "ewma":
        rp = inv_ewma(p, raw_root, raw_split, target, seq_len, trans_root, trans_split)
    else:
        raise ValueError(op)
    raw_t = _raw_windows(raw_root, raw_split, target, seq_len, h)      # (B,H) exact raw truth
    if rp.shape != raw_t.shape:
        raise ValueError(f"shape mismatch pred {rp.shape} vs raw truth {raw_t.shape}")
    # sanity: restore the transformed TRUE and confirm it reproduces the raw truth
    true = np.load(Path(pred_dir) / "true.npy")
    tt = true[:, :, 0] if true.ndim == 3 else true
    if op == "diff":
        rt = inv_diff(tt, raw_root, raw_split, target, seq_len)
    elif op == "diff2":
        rt = inv_diff2(tt, raw_root, raw_split, target, seq_len)
    elif op == "cusum":
        rt = inv_cusum(tt, raw_root, raw_split, target, seq_len, trans_root, trans_split)
    else:
        rt = inv_ewma(tt, raw_root, raw_split, target, seq_len, trans_root, trans_split)
    err = float(np.abs(rt - raw_t).max())
    if err >= 1e-3:
        raise AssertionError(f"[{op}] restore sanity FAIL: max|restored_true - raw_truth|={err:.3e}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred.npy", rp[:, :, None].astype(np.float32))
    np.save(out / "true.npy", raw_t[:, :, None].astype(np.float32))
    (out / "meta.json").write_text(f'{{"method": "op-{op}", "restore_true_err": {err:.3e}}}\n',
                                   encoding="utf-8")
    print(f"[{op}] restored {rp.shape} -> {out}  (sanity max_err={err:.2e} OK)")


def build_union(expert_dirs, low, high, out_dir, base=0):
    """Per-step max/min envelope union: pred crosses iff ANY expert crosses.
    max tests high, min tests low; non-crossing steps default to base expert."""
    preds = []
    for d in expert_dirs:
        p = np.load(Path(d) / "pred.npy"); preds.append(p[:, :, 0] if p.ndim == 3 else p)
    P = np.stack(preds, 0)                       # (K,B,H)
    hi, lo = P.max(0), P.min(0)
    out = preds[base].copy()
    out = np.where(hi > high, hi, out)
    out = np.where(lo < low, lo, out)
    o = Path(out_dir); o.mkdir(parents=True, exist_ok=True)
    bt = np.load(Path(expert_dirs[base]) / "true.npy"); bt = bt[:, :, 0] if bt.ndim == 3 else bt
    np.save(o / "pred.npy", out[:, :, None].astype(np.float32))
    np.save(o / "true.npy", bt[:, :, None].astype(np.float32))
    (o / "meta.json").write_text('{"method": "union-max-min"}\n', encoding="utf-8")
    print(f"[union] {P.shape[0]} experts -> {o}")


# ---------- CLIs ----------
def _preprocess(op, src_dir, dst_dir, glob):
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob(glob))
    if not paths:
        raise RuntimeError(f"no files matching {glob} in {src}")
    for path in paths:
        frame = pd.read_csv(path)
        has_time = frame.columns[0] == "Time"
        cols = frame.columns[1:] if has_time else frame.columns
        vals = frame[cols].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(vals).all():
            raise ValueError(f"non-finite values in {path}")
        out = frame.copy()
        out[cols] = FORWARD[op](vals)
        out.to_csv(dst / path.name, index=False)
    print(f"[{op}] wrote {len(paths)} files -> {dst}")


def _selftest():
    rng = np.random.RandomState(0)
    x = np.cumsum(rng.randn(300, 3), axis=0) + 5.0
    for op in FORWARD:
        t = FORWARD[op](x)
        # full-series round trip on the target (col 0), horizon = whole tail
        if op == "diff":
            r = x[0, 0] + np.cumsum(t[1:, 0])
            base = x[1:, 0]
        elif op == "diff2":
            d = (x[1, 0] - x[0, 0]) + np.cumsum(t[2:, 0])
            r = x[1, 0] + np.cumsum(d)
            base = x[2:, 0]
        elif op == "cusum":
            mu = x[:MU_PREONSET_ROWS, 0].mean() if len(x) >= MU_PREONSET_ROWS else x[:, 0].mean()
            r = mu + np.diff(t[:, 0])
            base = x[1:, 0]
        else:  # ewma
            a = EWMA_ALPHA
            r = (t[1:, 0] - (1 - a) * t[:-1, 0]) / a
            base = x[1:, 0]
        err = np.abs(r - base).max()
        print(f"[selftest] {op:6s} max|round-trip - raw| = {err:.2e}  {'OK' if err < 1e-6 else 'FAIL'}")


def _golden_diff(result_dir, root, raw_split, target, seq_len):
    rd = Path(result_dir)
    pred_diff = np.load(rd / "pred_diff.npy")
    pred_raw = np.load(rd / "pred.npy")
    pd2 = pred_diff[:, :, 0] if pred_diff.ndim == 3 else pred_diff
    pr2 = pred_raw[:, :, 0] if pred_raw.ndim == 3 else pred_raw
    restored = inv_diff(pd2, root, raw_split, target, seq_len)
    err = np.abs(restored - pr2).max()
    print(f"[golden diff] shape={pd2.shape} max|offline - in-exp restore| = {err:.2e}  "
          f"{'OK' if err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("preprocess")
    pp.add_argument("--op", required=True, choices=list(FORWARD))
    pp.add_argument("--src", required=True); pp.add_argument("--dst", required=True)
    pp.add_argument("--glob", default="*IDV13*Run*.csv")
    mv = sub.add_parser("multiview")
    mv.add_argument("--src", required=True); mv.add_argument("--dst", required=True)
    mv.add_argument("--target", required=True); mv.add_argument("--glob", default="*IDV13*Run*.csv")
    sub.add_parser("selftest")
    gd = sub.add_parser("golden-diff")
    gd.add_argument("--result-dir", required=True); gd.add_argument("--root", required=True)
    gd.add_argument("--raw-split", required=True); gd.add_argument("--target", required=True)
    gd.add_argument("--seq-len", type=int, default=96)
    rr = sub.add_parser("restore-result")
    rr.add_argument("--op", required=True, choices=list(FORWARD))
    rr.add_argument("--pred-dir", required=True)
    rr.add_argument("--raw-root", required=True); rr.add_argument("--raw-split", required=True)
    rr.add_argument("--trans-root", required=True); rr.add_argument("--trans-split", required=True)
    rr.add_argument("--target", required=True); rr.add_argument("--seq-len", type=int, default=96)
    rr.add_argument("--out-dir", required=True)
    un = sub.add_parser("union")
    un.add_argument("--expert", action="append", required=True, help="expert result dir (repeatable)")
    un.add_argument("--low", type=float, required=True); un.add_argument("--high", type=float, required=True)
    un.add_argument("--base", type=int, default=0); un.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "preprocess":
        _preprocess(a.op, a.src, a.dst, a.glob)
    elif a.cmd == "multiview":
        build_multiview(a.src, a.dst, a.target, a.glob)
    elif a.cmd == "selftest":
        _selftest()
    elif a.cmd == "golden-diff":
        _golden_diff(a.result_dir, a.root, a.raw_split, a.target, a.seq_len)
    elif a.cmd == "restore-result":
        restore_result(a.op, a.pred_dir, a.raw_root, a.raw_split, a.trans_root,
                       a.trans_split, a.target, a.seq_len, a.out_dir)
    elif a.cmd == "union":
        build_union(a.expert, a.low, a.high, a.out_dir, a.base)
