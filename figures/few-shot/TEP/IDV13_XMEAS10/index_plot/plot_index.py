"""Per-window forecast comparison: prediction curves vs ground truth at a chosen index.

For a chosen test-window index t (0..3619, matching the row order of every
pred.npy under results/), the figure shows:

- the last 30 true context samples left of the forecast origin (solid ink)
- the 15-step true future right of the origin (dashed ink, open markers)
- each selected method's 15-step prediction (colored lines)

Window t maps onto the raw CSVs as: Run9 for t < 1810 / Run10 for t >= 1810,
horizon starting at csv row 96 + (t % 1810).  Sampling is 0.05 h = 3 min per
step, so the x-axis is minutes relative to the forecast origin (-90 .. +45).

Usage (tsfm env):
    /home/aicode/miniconda3/envs/tsfm/bin/python plot_index.py --index 100
    /home/aicode/miniconda3/envs/tsfm/bin/python plot_index.py -i 100 --ratio r0p01
    /home/aicode/miniconda3/envs/tsfm/bin/python plot_index.py -i 700 --models all
    /home/aicode/miniconda3/envs/tsfm/bin/python plot_index.py -i 100 --models gate,diff,TCNTransformer_diff

--models presets:
    main (default) : gate + 4 TSFM experts + best two DL baselines (diff domain)
    tsfm           : gate + 4 TSFM experts
    all            : all 15 methods
or a comma list of keys: gate, diff, raw, time_moe, sundial,
    {CNNLSTM,DiPCALSTM,LSTMGRU,STAConvBiLSTM,TCNTransformer}_{raw,diff}

Outputs pred_idx{index}_{ratio}.png (300 dpi) and .pdf next to this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("/home/aicode/sherwin/TSFM/results")
CSV_DIR = Path("/home/aicode/sherwin/dataset/TEP/csv_5var")

TARGET_COL = "XMEAS10 Purge Rate"
TRAIN_RUNS = range(1, 8)   # Run1-7: pre-onset data defines the +-3 sigma alarm limits
CONTEXT_SHOW = 30          # true samples shown left of the origin
HORIZON = 15               # fused / evaluated forecast steps (45 min)
SEQ_LEN = 96               # model context length -> horizon of window t starts at csv row 96+t
WINDOWS_PER_RUN = 1810     # test = Run9 (0..1809) + Run10 (1810..3619)
STEP_MIN = 3.0             # 0.05 h sampling
FAULT_ROW = 600            # IDV13 injected at 30 h = row 600 of each run

IKB = "#002FA7"
INK = "#0a0a0a"
GREY_DARK = "#5c5c5c"
GREY_MID = "#8f8f8d"
GREY_LIGHT = "#c4c4c2"
PAPER = "#ffffff"

DL_MODELS = ["CNNLSTM", "DiPCALSTM", "LSTMGRU", "STAConvBiLSTM", "TCNTransformer"]
DL_PALETTE = {
    "CNNLSTM": "#b0803c",
    "DiPCALSTM": "#7a9a6d",
    "LSTMGRU": "#9a6d8f",
    "STAConvBiLSTM": "#3c7f8f",
    "TCNTransformer": "#c25b4e",
}

PRESETS = {
    "main": ["gate", "diff", "raw", "time_moe", "sundial",
             "TCNTransformer_diff", "STAConvBiLSTM_diff"],
    "tsfm": ["gate", "diff", "raw", "time_moe", "sundial"],
    "all": ["gate", "diff", "raw", "time_moe", "sundial"]
           + [f"{m}_{d}" for m in DL_MODELS for d in ("diff", "raw")],
}


def result_dir(key: str, ratio: str) -> Path:
    """Resolve the results/ folder holding pred.npy for a method key."""
    if key == "gate":
        return RESULTS / f"ensemble_Gate_XMEAS10_{ratio}_test"
    if key == "time_moe":
        return RESULTS / "fm_time_moe_xmeas10_zeroshot_test"       # ratio-independent
    if key == "sundial":
        return RESULTS / "fm_sundial_xmeas10_zeroshot_test"        # ratio-independent
    if key in ("diff", "raw"):
        pat = f"forecast_TEP_IDV13_XMEAS10_5var_{key}_few_{ratio}_timer_xl_*"
    else:
        model, dom = key.rsplit("_", 1)
        pat = f"long_term_forecast_TEP_IDV13_XMEAS10_5var_{dom}_{model}_{ratio}_{model}_*"
    hits = sorted(RESULTS.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no results folder for {key} @ {ratio} (pattern {pat})")
    return hits[0]


def load_pred(key: str, ratio: str, index: int) -> np.ndarray:
    pred = np.load(result_dir(key, ratio) / "pred.npy")
    return np.asarray(pred[index]).reshape(-1)[:HORIZON].astype(float)


def load_truth_around(index: int) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Return (context_tail, true_future, origin_hour, run_name) from the raw CSV."""
    run_no, local = (9, index) if index < WINDOWS_PER_RUN else (10, index - WINDOWS_PER_RUN)
    hits = sorted(CSV_DIR.glob(f"*IDV13*Run{run_no}.csv"))
    if not hits:
        raise FileNotFoundError(f"no csv for Run{run_no} in {CSV_DIR}")
    series = pd.read_csv(hits[0])[TARGET_COL].to_numpy(float)
    start = SEQ_LEN + local                      # csv row where the horizon begins
    ctx = series[max(0, start - CONTEXT_SHOW):start]
    fut = series[start:start + HORIZON]
    return ctx, fut, start * 0.05, f"Run{run_no}"


def control_limits() -> tuple[float, float]:
    """+-3 sigma limits from pre-onset (t < 30 h) data of train Run1-7,
    same convention as the event-recall evaluation."""
    chunks = []
    for n in TRAIN_RUNS:
        series = pd.read_csv(sorted(CSV_DIR.glob(f"*IDV13*Run{n}.csv"))[0])[TARGET_COL]
        chunks.append(series.to_numpy(float)[:FAULT_ROW])
    v = np.concatenate(chunks)
    return float(v.mean() - 3 * v.std()), float(v.mean() + 3 * v.std())


def style_for(key: str) -> dict:
    if key == "gate":
        return dict(color=IKB, lw=2.6, marker="o", ms=5.0, mfc=IKB, mec="white",
                    mew=1.0, zorder=8, label="Gate fusion")
    if key == "diff":
        return dict(color=INK, lw=1.7, marker="o", ms=3.8, mfc=PAPER, mew=1.1,
                    zorder=6, label="Timer-XL · diff")
    if key == "raw":
        return dict(color=GREY_DARK, lw=1.7, marker="o", ms=3.8, mfc=PAPER, mew=1.1,
                    zorder=5, label="Timer-XL · raw")
    if key == "time_moe":
        return dict(color=GREY_MID, lw=1.5, ls=(0, (4, 3)), zorder=4,
                    label="Time-MoE · zero-shot")
    if key == "sundial":
        return dict(color=GREY_MID, lw=1.5, ls=(0, (1, 2.2)), zorder=4,
                    label="Sundial · zero-shot")
    model, dom = key.rsplit("_", 1)
    return dict(color=DL_PALETTE[model], lw=1.3, alpha=0.9,
                ls="-" if dom == "diff" else (0, (5, 2)), zorder=3,
                label=f"{model} · {dom}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--index", type=int, default=100,
                    help="test-window index, 0..3619 (default 100)")
    ap.add_argument("--ratio", default="r1p0",
                    help="few-shot ratio tag: r0p01 r0p02 r0p05 r0p1 r0p25 r0p5 r1p0")
    ap.add_argument("--models", default="main",
                    help="preset (main/tsfm/all) or comma list of method keys")
    ap.add_argument("--limits", default=True, action=argparse.BooleanOptionalAction,
                    help="draw the +-3 sigma alarm limits (LCL/UCL); --no-limits hides")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                    help="zoom the y-axis; a clipped true future gets annotated")
    args = ap.parse_args()

    if not 0 <= args.index < 2 * WINDOWS_PER_RUN:
        raise SystemExit(f"--index must be in [0, {2 * WINDOWS_PER_RUN - 1}]")
    keys = PRESETS.get(args.models, [k.strip() for k in args.models.split(",") if k.strip()])

    ctx, fut, origin_h, run = load_truth_around(args.index)
    preds = {k: load_pred(k, args.ratio, args.index) for k in keys}
    # cross-check saved ground truth against the raw csv
    ref = np.load(result_dir("gate", args.ratio) / "true.npy")[args.index].reshape(-1)[:HORIZON]
    assert np.allclose(ref, fut, atol=1e-5), "true.npy does not match csv alignment"

    t_ctx = (np.arange(-len(ctx), 0) + 1) * STEP_MIN    # origin = last context point at 0 min
    t_fut = np.arange(1, HORIZON + 1) * STEP_MIN

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.grid(axis="y", color=GREY_LIGHT, lw=0.6, alpha=0.55)
    ax.tick_params(colors=GREY_DARK, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GREY_LIGHT)

    # regions: observed past (grey) vs forecast horizon (IKB tint)
    ax.axvspan(t_ctx[0] - STEP_MIN, 0, color=GREY_LIGHT, alpha=0.18, zorder=0)
    ax.axvspan(0, t_fut[-1] + STEP_MIN * 0.5, color=IKB, alpha=0.045, zorder=0)
    ax.axvline(0, color=GREY_DARK, lw=0.9, ls=(0, (2, 2)), zorder=1)

    # ground truth
    ax.plot(t_ctx, ctx, color=INK, lw=1.9, marker="o", ms=3.6, zorder=7,
            label="True · context")
    ax.plot(np.r_[0, t_fut], np.r_[ctx[-1], fut], color=INK, lw=1.9,
            ls=(0, (4, 2.4)), marker="o", ms=4.6, mfc=PAPER, mew=1.3, zorder=7,
            label="True · future")

    # predictions, best first so the legend ranks methods for this window
    mses = {k: float(np.mean((preds[k] - fut) ** 2)) for k in keys}
    for k in sorted(keys, key=mses.get):
        st = style_for(k)
        st["label"] = f"{st['label']}   {mses[k] * 1e3:.2f}"
        ax.plot(np.r_[0, t_fut], np.r_[ctx[-1], preds[k]], **st)

    # alarm limits: drawn after the curves, ylim restored so they never distort
    # the view when a limit sits far outside the data range
    if args.limits:
        lo_lim, hi_lim = control_limits()
        ax.relim()
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        for y_lim, name in ((lo_lim, "LCL"), (hi_lim, "UCL")):
            ax.axhline(y_lim, color="#a03a2e", lw=1.0, ls=(0, (6, 3)), alpha=0.8,
                       zorder=2)
            if y0 <= y_lim <= y1:
                ax.text(t_ctx[0], y_lim, f"{name} $\\mu{'-' if name == 'LCL' else '+'}3\\sigma$",
                        fontsize=8.2, color="#a03a2e", va="bottom", ha="left",
                        fontfamily="monospace")
        ax.set_ylim(y0, y1)

    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
        below, above = fut < args.ylim[0], fut > args.ylim[1]
        for mask, edge, arrow in ((below, args.ylim[0], "↓"), (above, args.ylim[1], "↑")):
            if mask.any():
                x_exit = t_fut[np.argmax(mask)]
                extreme = fut.min() if arrow == "↓" else fut.max()
                ax.annotate(f"{arrow} true reaches {extreme:.3f}",
                            xy=(x_exit, edge), xytext=(6, 10 if arrow == "↓" else -14),
                            textcoords="offset points", fontsize=8.8, color=INK,
                            fontweight="bold")

    ax.set_xlim(t_ctx[0] - STEP_MIN, t_fut[-1] + STEP_MIN * 0.5)
    ax.set_xlabel("time relative to forecast origin (min)", fontsize=10.5,
                  color=GREY_DARK, labelpad=8)
    ax.set_ylabel("XMEAS10 Purge Rate (kscmh)", fontsize=10.5, color=GREY_DARK)

    ax.text(t_ctx[0], 1.015, "OBSERVED CONTEXT (LAST 30 SAMPLES)", fontsize=8.2,
            color=GREY_DARK, fontfamily="monospace", transform=ax.get_xaxis_transform())
    ax.text(t_fut[-1], 1.015, "FORECAST · 15 STEPS = 45 MIN", fontsize=8.2,
            color=IKB, fontweight="bold", fontfamily="monospace", ha="right",
            transform=ax.get_xaxis_transform())

    fault_state = ("post-fault" if origin_h >= FAULT_ROW * 0.05 else "pre-fault")
    leg = ax.legend(loc="best", fontsize=8.8, frameon=False, labelcolor="linecolor",
                    title=r"MSE $\times 10^{-3}$ @ this window", title_fontsize=8.4,
                    alignment="left")
    leg.get_title().set_color(GREY_DARK)

    fig.subplots_adjust(left=0.085, right=0.965, top=0.945, bottom=0.13)
    bar = plt.Rectangle((0.085, 0.945), 0.045, 0.012, transform=fig.transFigure,
                        facecolor=IKB, edgecolor="none")
    # fig.add_artist(bar)
    # fig.text(0.085, 0.895, f"Window #{args.index} · {run} · origin t = {origin_h:.2f} h "
    #                        f"({fault_state}, IDV13 @ 30 h) · data ratio {args.ratio}",
    #          fontsize=13, color=INK, fontweight="bold", va="bottom")
    fig.text(0.085, 0.028, "TEP IDV13 · XMEAS10 purge rate",
             fontsize=8, color=GREY_MID, va="bottom")

    stem = f"pred_idx{args.index}_{args.ratio}"
    fig.savefig(HERE / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(HERE / f"{stem}.pdf", facecolor=fig.get_facecolor())
    print(f"saved {HERE / stem}.png and .pdf")
    for k in sorted(keys, key=mses.get):
        print(f"  {k:24s} MSE@15 = {mses[k] * 1e3:8.3f} x1e-3")


if __name__ == "__main__":
    main()
