"""Full-data vs few-shot forecast, ONE method, on a single test window.

Draws the prediction-vs-truth curves for one method trained on the FULL data
(ratio r1p0) and on FEW-SHOT data (ratio r0p01), side by side and as two
standalone panels, in English and/or Chinese.

Default method is STAConvBiLSTM_raw -- the DL baseline with the largest
full->few MSE gap (16.3 -> 54.7 x1e-3 aggregate, 3.36x). Default window #575 is a
real IDV13 fault excursion (XMEAS10 spikes to ~0.8); with 1% data the raw
baseline collapses to the global mean and misses the surge entirely.

Run (tsfm env has numpy/pandas/matplotlib):
    PY=/home/aicode/miniconda3/envs/tsfm/bin/python
    $PY plot_full_vs_few.py                        # defaults: idx 575, STAConvBiLSTM_raw, both langs
    $PY plot_full_vs_few.py --index 528 --lang en
    $PY plot_full_vs_few.py --method TCNTransformer_raw --index 700
    $PY plot_full_vs_few.py --full-ratio r1p0 --few-ratio r0p05

--method keys:
    {CNNLSTM,DiPCALSTM,LSTMGRU,STAConvBiLSTM,TCNTransformer}_{raw,diff}
    or a TSFM expert: gate, diff, raw, time_moe, sundial
--ratio tags: r0p01 r0p02 r0p05 r0p1 r0p25 r0p5 r1p0

Outputs (next to this script): <method>_{full,few,compare}_idx<N>[_en].png + .pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = Path("/home/aicode/sherwin/TSFM/results")
CSV_DIR = Path("/home/aicode/sherwin/dataset/TEP/csv_5var")
TARGET = "XMEAS10 Purge Rate"

H, SEQ, CTX_SHOW, STEP = 15, 96, 30, 3.0      # horizon, context len, context shown, min/step
WPR = 1810                                     # windows per run: Run9 [0..1809], Run10 [1810..]
FAULT_ROW = 600                                # IDV13 injected at 30 h = csv row 600
IKB, INK, TEAL, GREY_L, GREY_D = "#002FA7", "#0a0a0a", "#2f6f7e", "#c4c4c2", "#5c5c5c"
AMBER, RED = "#8a5a00", "#a03a2e"

TEXT = {
    "en": dict(
        ylabel="XMEAS10 Purge Rate (kscmh)", xlabel="time relative to forecast origin (min)",
        ctx="True · context", fut="True · future (45 min)", ucl="  alarm limit  UCL",
        full_title="Full data (100%)", few_title="Few-shot ({few}% of training data)",
        full_badge="FULL · 100%", few_badge="FEW-SHOT · {few}%",
        pred="Traditional Deep Learning · forecast",
        annot="forecast ≈ {m:.2f} (near-constant),\nmisses the fault surge",
        suptitle="Same fault window, same model — forecast gap as training data drops from 100% to {few}%",
        foot="{name} · TEP fault IDV13 · XMEAS10 Purge Rate · test window #{idx} (Run{run}, origin t={oh:.1f} h, post-fault)",
        foot1="TEP IDV13 · window #{idx} · Run{run} · post-fault t={oh:.1f} h",
        fonts=["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"], suffix="_en"),
    "zh": dict(
        ylabel="XMEAS10 吹扫流量 Purge Rate (kscmh)", xlabel="相对预测起点的时间 (min)",
        ctx="真实值 · 历史", fut="真实值 · 未来 45 min", ucl="  报警上限 UCL",
        full_title="全量数据训练 (100%)", few_title="少量数据训练 ({few}% · few-shot)",
        full_badge="全量 100%", few_badge="少量 {few}%",
        pred="{name} 预测",
        annot="预测坍塌至均值 ≈ {m:.2f}\n完全错过故障骤升",
        suptitle="同一故障窗口，同一模型 —— 训练数据从 100% 降到 {few}% 的预测差距",
        foot="{name} · TEP 故障 IDV13 · XMEAS10 吹扫流量 · 测试窗口 #{idx} (Run{run}, 起点 t={oh:.1f} h, 故障后)",
        foot1="TEP IDV13 · 窗口 #{idx} · Run{run} · 故障后 t={oh:.1f} h",
        fonts=["Noto Sans SC", "Inter", "DejaVu Sans"], suffix=""),
}

EXPERT_NAMES = {"gate": "Gate fusion", "diff": "Timer-XL · diff", "raw": "Timer-XL · raw",
                "time_moe": "Time-MoE", "sundial": "Sundial"}


def result_dir(key: str, ratio: str) -> Path:
    if key == "gate":
        return RESULTS / f"ensemble_Gate_XMEAS10_{ratio}_test"
    if key == "time_moe":
        return RESULTS / "fm_time_moe_xmeas10_zeroshot_test"
    if key == "sundial":
        return RESULTS / "fm_sundial_xmeas10_zeroshot_test"
    if key in ("diff", "raw"):
        pat = f"forecast_TEP_IDV13_XMEAS10_5var_{key}_few_{ratio}_timer_xl_*"
    else:
        model, dom = key.rsplit("_", 1)
        pat = f"long_term_forecast_TEP_IDV13_XMEAS10_5var_{dom}_{model}_{ratio}_{model}_*"
    hits = sorted(RESULTS.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no results folder for '{key}' @ {ratio} (pattern: {pat})")
    return hits[0]


def method_name(key: str) -> str:
    return EXPERT_NAMES.get(key, key.rsplit("_", 1)[0])


def load_pred(key: str, ratio: str, idx: int) -> np.ndarray:
    p = np.load(result_dir(key, ratio) / "pred.npy")
    return np.asarray(p[idx]).reshape(-1)[:H].astype(float)


def load_truth(idx: int):
    run, local = (9, idx) if idx < WPR else (10, idx - WPR)
    csv = sorted(CSV_DIR.glob(f"*IDV13*Run{run}.csv"))[0]
    s = pd.read_csv(csv)[TARGET].to_numpy(float)
    start = SEQ + local
    return s[start - CTX_SHOW:start], s[start:start + H], start * 0.05, run


def control_limits():
    ch = []
    for n in range(1, 8):
        s = pd.read_csv(sorted(CSV_DIR.glob(f"*IDV13*Run{n}.csv"))[0])[TARGET].to_numpy(float)
        ch.append(s[:FAULT_ROW])
    v = np.concatenate(ch)
    return float(v.mean() - 3 * v.std()), float(v.mean() + 3 * v.std())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--index", type=int, default=575, help="test-window index 0..3619")
    ap.add_argument("--method", default="STAConvBiLSTM_raw", help="method key (see header)")
    ap.add_argument("--full-ratio", default="r1p0")
    ap.add_argument("--few-ratio", default="r0p01")
    ap.add_argument("--lang", default="both", choices=["en", "zh", "both"])
    ap.add_argument("--no-annotate", action="store_true", help="hide the few-shot annotation")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    idx = args.index
    ctx, fut, origin_h, run = load_truth(idx)
    pf = load_pred(args.method, args.full_ratio, idx)
    pw = load_pred(args.method, args.few_ratio, idx)
    # sanity: saved ground truth must match the csv window (alignment guard)
    ref = np.load(result_dir(args.method, args.full_ratio) / "true.npy")[idx].reshape(-1)[:H]
    assert np.allclose(ref, fut, atol=1e-4), "true.npy does not match csv alignment for this index"

    mse_f = float(np.mean((pf - fut) ** 2)) * 1e3
    mse_w = float(np.mean((pw - fut) ** 2)) * 1e3
    lo, hi = control_limits()
    few_pct = float(args.few_ratio.replace("r", "").replace("p", ".")) * 100
    few_pct = f"{few_pct:g}"
    name = method_name(args.method)

    t_ctx = (np.arange(-len(ctx), 0) + 1) * STEP
    t_fut = np.arange(1, H + 1) * STEP
    allv = np.concatenate([ctx, fut, pf, pw, [hi]])
    pad = (allv.max() - allv.min()) * 0.08
    ylim = (allv.min() - pad, allv.max() + pad + 0.10)
    print(f"idx {idx} Run{run} origin {origin_h:.2f}h  {args.method}  MSE full {mse_f:.1f}  few {mse_w:.1f} (x1e-3)")

    langs = ["en", "zh"] if args.lang == "both" else [args.lang]
    for lang in langs:
        T = TEXT[lang]
        plt.rcParams.update({"font.sans-serif": T["fonts"], "axes.unicode_minus": False,
                             "axes.linewidth": 0.8, "svg.fonttype": "none"})

        def draw(ax, pred, mse, title, badge, badge_color):
            ax.set_facecolor("#ffffff")
            ax.grid(axis="y", color=GREY_L, lw=0.6, alpha=0.55)
            ax.tick_params(colors=GREY_D, labelsize=10)
            for sp in ax.spines.values():
                sp.set_color(GREY_L)
            ax.axvspan(t_ctx[0] - STEP, 0, color=GREY_L, alpha=0.18, zorder=0)
            ax.axvspan(0, t_fut[-1] + STEP * 0.5, color=IKB, alpha=0.05, zorder=0)
            ax.axvline(0, color=GREY_D, lw=0.9, ls=(0, (2, 2)), zorder=1)
            ax.axhline(hi, color=RED, lw=1.0, ls=(0, (6, 3)), alpha=0.85, zorder=2)
            if ylim[0] <= hi <= ylim[1]:
                ax.text(t_ctx[0], hi, T["ucl"] + r" $\mu+3\sigma$", fontsize=8.6, color=RED,
                        va="bottom", ha="left")
            ax.plot(t_ctx, ctx, color=INK, lw=1.9, marker="o", ms=3.4, zorder=7, label=T["ctx"])
            ax.plot(np.r_[0, t_fut], np.r_[ctx[-1], fut], color=INK, lw=2.0, ls=(0, (4, 2.2)),
                    marker="o", ms=4.6, mfc="#ffffff", mew=1.3, zorder=7, label=T["fut"])
            ax.plot(np.r_[0, t_fut], np.r_[ctx[-1], pred], color=TEAL, lw=2.6, marker="s",
                    ms=4.2, mfc=TEAL, mec="white", mew=0.8, zorder=6,
                    label=T["pred"].format(name=name))
            ax.set_ylim(*ylim)
            ax.set_xlim(t_ctx[0] - STEP, t_fut[-1] + STEP * 0.5)
            ax.set_xlabel(T["xlabel"], fontsize=10.5, color=GREY_D, labelpad=7)
            ax.text(0.028, 0.955, badge, transform=ax.transAxes, fontsize=12.5, fontweight="bold",
                    color="#ffffff", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc=badge_color, ec="none"))
            ax.text(0.028, 0.845, "MSE = %.1f " % mse + r"$\times10^{-3}$", transform=ax.transAxes,
                    fontsize=11.5, color=INK, va="top", ha="left", fontweight="bold")
            ax.set_title(title, fontsize=13.5, color=INK, fontweight="bold", pad=8)
            ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor="linecolor")

        def annotate_few(ax):
            if args.no_annotate:
                return
            ax.annotate(T["annot"].format(m=float(pw.mean())), xy=(t_fut[9], pw[9]),
                        xytext=(0.45, 0.41), textcoords="axes fraction", fontsize=9.6,
                        color=AMBER, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3))

        # combined
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharey=True)
        draw(axes[0], pf, mse_f, T["full_title"], T["full_badge"], IKB)
        draw(axes[1], pw, mse_w, T["few_title"].format(few=few_pct), T["few_badge"].format(few=few_pct), AMBER)
        axes[0].set_ylabel(T["ylabel"], fontsize=10.8, color=GREY_D)
        annotate_few(axes[1])
        fig.suptitle(T["suptitle"].format(few=few_pct), fontsize=15, fontweight="bold", color=INK, y=0.995)
        fig.text(0.5, 0.012, T["foot"].format(name=name, idx=idx, run=run, oh=origin_h),
                 fontsize=8.6, color="#8f8f8d", ha="center")
        fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.11, wspace=0.06)
        stem = f"{args.method}_compare_idx{idx}{T['suffix']}"
        fig.savefig(args.outdir / f"{stem}.png", dpi=300, facecolor="#ffffff")
        fig.savefig(args.outdir / f"{stem}.pdf", facecolor="#ffffff")
        plt.close(fig)

        # standalone
        for pred, mse, title, badge, color, tag, annot in [
            (pf, mse_f, T["full_title"], T["full_badge"], IKB, "full", False),
            (pw, mse_w, T["few_title"].format(few=few_pct), T["few_badge"].format(few=few_pct), AMBER, "few", True),
        ]:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            draw(ax, pred, mse, title, badge, color)
            ax.set_ylabel(T["ylabel"], fontsize=10.5, color=GREY_D)
            if annot:
                annotate_few(ax)
            fig.text(0.085, 0.017, T["foot1"].format(idx=idx, run=run, oh=origin_h),
                     fontsize=8, color="#8f8f8d")
            fig.subplots_adjust(left=0.1, right=0.965, top=0.9, bottom=0.12)
            stem = f"{args.method}_{tag}_idx{idx}{T['suffix']}"
            fig.savefig(args.outdir / f"{stem}.png", dpi=300, facecolor="#ffffff")
            fig.savefig(args.outdir / f"{stem}.pdf", facecolor="#ffffff")
            plt.close(fig)

    print(f"saved {args.method} full/few/compare figures ({', '.join(langs)}) -> {args.outdir}")


if __name__ == "__main__":
    main()
