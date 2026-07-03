"""Few-shot figures for TEP IDV13 / XMEAS07.

Fig 1: bar chart of MSE (h=15) vs few-shot fraction, raw / DIFF / Gate-T2.
Fig 2: at one test instant, the Gate-T2 forecast under each few-shot fraction,
       overlaid on the true trajectory (absolute reactor-pressure scale).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path("/home/aicode/sherwin/TSFM")
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "TEP_IDV13_XMEAS07_FewShot_Summary"
DATA_ROOT = Path("/home/aicode/sherwin/dataset/TEP")
TARGET = "XMEAS07 Reactor Pressure"
SPLIT = ROOT / "setting/TEP_IDV13_XMEAS07.yaml"   # test = Run9, Run10
FIGDIR = ROOT / "figures/few-shot/TEP/IDV13"
SEQ, PRED = 96, 96
ONSET_H = 30.0


def tag_of(ratio: float) -> str:
    return "r" + repr(ratio).replace(".", "p")


def gate_dir(ratio: float) -> Path:
    return RESULTS / f"ensemble_Gate-T2-TEP-IDV13-XMEAS07-S_few_{tag_of(ratio)}_test_0"


def view(arr: np.ndarray) -> np.ndarray:
    return arr[:, :, 0] if arr.ndim == 3 else arr


def control_limits() -> tuple[float, float]:
    cfg = yaml.safe_load(SPLIT.read_text())
    chunks = []
    for rel in cfg["train"]:
        df = pd.read_csv(DATA_ROOT / rel)
        chunks.append(df.loc[df["Time"] < ONSET_H, TARGET].to_numpy(float))
    v = np.concatenate(chunks)
    return float(v.mean() - 3 * v.std()), float(v.mean() + 3 * v.std())


def test_files() -> list[str]:
    return yaml.safe_load(SPLIT.read_text())["test"]


def locate(global_idx: int, usable: int):
    """global window index -> (file position, local index)."""
    return divmod(global_idx, usable)


def fig1_bar(ratios: list[float]) -> None:
    curve = {row["ratio"]: row for row in json.loads((SUMMARY / "curve.json").read_text())}
    rs = [r for r in ratios if r in curve]
    labels = [f"{r*100:g}%\n({curve[r]['n_train']})" for r in rs]
    raw = [curve[r]["raw_mse"] for r in rs]
    diff = [curve[r]["diff_mse"] for r in rs]
    gate = [curve[r]["gate_t2_mse"] for r in rs]

    x = np.arange(len(rs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, raw, w, label="raw expert", color="#9ecae1")
    ax.bar(x, diff, w, label="DIFF expert", color="#fdae6b")
    ax.bar(x + w, gate, w, label="Gate-T2 ensemble", color="#31a354")
    for xi, g in zip(x, gate):
        ax.text(xi + w, g + 1.5, f"{g:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("few-shot fraction (number of training windows)")
    ax.set_ylabel("MSE  (horizon = 15 steps, 45 min)")
    ax.set_title("Few-shot forecast MSE on TEP IDV13 / XMEAS07 (test = Run9-10)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "fig1_mse_vs_ratio_bar.png"
    fig.savefig(out, dpi=150)
    print(f"fig1 -> {out}")


def pick_window(low: float, high: float, usable: int, horizon: int) -> int:
    """First window in Run9 whose TRUE forecast crosses a control limit
    at/after onset — the earliest prognosis-relevant instant."""
    df = pd.read_csv(DATA_ROOT / test_files()[0])
    vals = df[TARGET].to_numpy(float)
    times = df["Time"].to_numpy(float)
    for local in range(usable):
        start = local + SEQ
        fut = vals[start:start + horizon]
        fut_t = times[start:start + horizon]
        crosses = (fut < low) | (fut > high)
        if crosses.any() and fut_t[crosses][0] >= ONSET_H:
            return local  # global index == local for Run9 (file 0)
    return usable // 3


def fig2_instant(ratios: list[float], window_index: int | None, horizon: int) -> None:
    low, high = control_limits()
    files = test_files()
    usable = len(pd.read_csv(DATA_ROOT / files[0])) - SEQ - PRED + 1

    gidx = pick_window(low, high, usable, horizon) if window_index is None else window_index
    fpos, local = locate(gidx, usable)
    df = pd.read_csv(DATA_ROOT / files[fpos])
    vals = df[TARGET].to_numpy(float)
    times = df["Time"].to_numpy(float)
    start = local + SEQ

    ctx_n = 48
    ctx_t = times[start - ctx_n:start]
    ctx_v = vals[start - ctx_n:start]
    fut_t = times[start:start + horizon]
    fut_v = vals[start:start + horizon]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(ctx_t, ctx_v, color="0.4", lw=1.5, label="context (look-back)")
    ax.plot(fut_t, fut_v, color="black", lw=2.6, label="ground truth", zorder=6)

    colors = cm.viridis(np.linspace(0.12, 0.92, len(ratios)))
    for ratio, c in zip(ratios, colors):
        gd = gate_dir(ratio)
        if not (gd / "pred.npy").exists():
            print(f"  (skip {ratio}: no pred at {gd})")
            continue
        pred = view(np.load(gd / "pred.npy"))[gidx, :horizon]
        ax.plot(fut_t, pred, "--", color=c, lw=1.8, marker="o", ms=3,
                label=f"Gate-T2 @ {ratio*100:g}%")

    ax.axvline(times[start], color="0.6", ls=":", lw=1)
    ax.axhline(low, color="tab:red", ls="--", lw=1, alpha=0.7)
    ax.axhline(high, color="tab:red", ls="--", lw=1, alpha=0.7, label="3σ control limits")
    ax.axvline(ONSET_H, color="tab:purple", ls="-.", lw=1.2, alpha=0.7, label="fault onset (30 h)")
    ax.set_xlabel("time (h)")
    ax.set_ylabel(TARGET)
    ax.set_title(f"Forecast at one test instant vs few-shot fraction "
                 f"({files[fpos].split('_')[-1].replace('.csv','')}, start={times[start]:.2f} h)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "fig2_forecast_instant_by_ratio.png"
    fig.savefig(out, dpi=150)
    print(f"fig2 -> {out}  (global window {gidx}, file {files[fpos]}, local {local})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--window-index", type=int, default=None,
                    help="global test window index for fig2 (default: first onset-crossing window)")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig1_bar(args.ratios)
    fig2_instant(args.ratios, args.window_index, args.horizon)


if __name__ == "__main__":
    main()
