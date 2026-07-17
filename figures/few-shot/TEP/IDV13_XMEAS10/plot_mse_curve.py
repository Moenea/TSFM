"""Presentation figure: MSE vs training-data ratio for gate / TSFM experts / DL baselines.

Reads results/TEP_IDV13_XMEAS10_Summary/metrics_r*.json (full precision, not the
rounded comparison.csv) and renders a Swiss-style two-panel (broken-y) curve:

- bottom panel: gate (IKB blue), Timer-XL diff/raw, zero-shot Time-MoE/Sundial,
  and the five diff-domain DL baselines (thin grey)
- top panel: the five raw-domain DL baselines, which collapse to ~mean (MSE ~ 50)

Usage:
    /home/aicode/miniconda3/envs/tsfm/bin/python plot_mse_curve.py

Outputs mse_vs_ratio.png (300 dpi) and mse_vs_ratio.pdf next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[3] / "results" / "TEP_IDV13_XMEAS10_Summary"

RATIOS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
RATIO_TAGS = ["r0p01", "r0p02", "r0p05", "r0p1", "r0p25", "r0p5", "r1p0"]
N_TRAIN = [126, 253, 633, 1267, 3167, 6335, 12670]

IKB = "#002FA7"
INK = "#0a0a0a"
GREY_DARK = "#5c5c5c"
GREY_MID = "#8f8f8d"
GREY_LIGHT = "#c4c4c2"
# PAPER = "#fafaf8"
PAPER = "#ffffff"

DL_MODELS = ["CNNLSTM", "DiPCALSTM", "LSTMGRU", "STAConvBiLSTM", "TCNTransformer"]
SCALE = 1e3  # plot MSE x 10^-3


def load_curves() -> dict[str, np.ndarray]:
    curves: dict[str, list[float]] = {}
    for tag in RATIO_TAGS:
        with (RESULTS / f"metrics_{tag}.json").open() as fh:
            payload = json.load(fh)
        for name, entry in payload.get("models", payload).items():
            curves.setdefault(name, []).append(entry["mse"] * SCALE)
    return {name: np.asarray(vals) for name, vals in curves.items()}


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions apart (ascending order preserved)."""
    order = np.argsort(ys)
    spread = list(ys)
    for prev, cur in zip(order[:-1], order[1:]):
        if spread[cur] - spread[prev] < min_gap:
            spread[cur] = spread[prev] + min_gap
    return spread


def main() -> None:
    curves = load_curves()
    x = np.asarray(RATIOS)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })

    fig, (ax_top, ax) = plt.subplots(
        2, 1, sharex=True, figsize=(9.3, 6.2),
        gridspec_kw={"height_ratios": [1, 4.2], "hspace": 0.06},
    )
    fig.patch.set_facecolor(PAPER)
    for a in (ax_top, ax):
        a.set_facecolor(PAPER)
        a.set_xscale("log")
        a.grid(axis="y", color=GREY_LIGHT, lw=0.6, alpha=0.55)
        a.tick_params(colors=GREY_DARK, labelsize=10)
        for spine in a.spines.values():
            spine.set_color(GREY_LIGHT)

    # ---- few-shot regime shading (ratio <= 0.1) -------------------------------
    for a in (ax_top, ax):
        a.axvspan(0.009, 0.1, color=IKB, alpha=0.045, zorder=0)

    # ---- raw-domain DL baselines: drawn on BOTH panels so lines that improve
    # at high ratios (STA/TCN raw) visibly cross the axis break ---------------
    for model in DL_MODELS:
        ax_top.plot(x, curves[f"{model}_raw"], color=GREY_LIGHT, lw=1.1, zorder=2)
        ax.plot(x, curves[f"{model}_raw"], color=GREY_LIGHT, lw=1.1, zorder=2)
    ax_top.set_ylim(40, 58)
    ax_top.set_yticks([45, 55])
    ax_top.text(
        1.03, 53.0, "DL baselines · raw",
        transform=ax_top.get_yaxis_transform(), fontsize=9.5, color=GREY_MID,
        va="center", ha="left",
    )

    # broken-axis marks
    ax_top.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    d = 0.007
    kwargs = dict(transform=ax_top.transAxes, color=GREY_DARK, clip_on=False, lw=0.9)
    ax_top.plot((-d, +d), (-0.06, +0.06), **kwargs)
    kwargs.update(transform=ax.transAxes)
    ax.plot((-d, +d), (1 - 0.015, 1 + 0.015), **kwargs)

    # ---- bottom panel ---------------------------------------------------------
    # diff-domain DL baselines: thin grey context lines
    for model in DL_MODELS:
        ax.plot(x, curves[f"{model}_diff"], color=GREY_LIGHT, lw=1.1, zorder=2)

    # zero-shot experts: constant dashed lines
    ax.plot(x, curves["time_moe"], color=GREY_MID, lw=1.4, ls=(0, (4, 3)), zorder=3)
    ax.plot(x, curves["sundial"], color=GREY_MID, lw=1.4, ls=(0, (1, 2.2)), zorder=3)

    # fine-tuned Timer-XL experts
    ax.plot(x, curves["raw"], color=GREY_DARK, lw=1.8, marker="o", ms=4.5,
            mfc=PAPER, mew=1.3, zorder=4)
    ax.plot(x, curves["diff"], color=INK, lw=1.8, marker="o", ms=4.5,
            mfc=PAPER, mew=1.3, zorder=4)

    # the hero: gate fusion
    ax.plot(x, curves["gate"], color=IKB, lw=3.2, marker="o", ms=7.5,
            mfc=IKB, mec="white", mew=1.6, zorder=6,
            solid_capstyle="round")

    ax.set_ylim(4, 19.6)
    ax.set_yticks([5, 7.5, 10, 12.5, 15, 17.5])
    ax.set_xlim(0.009, 1.12)
    ax.set_xticks(RATIOS)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(
        [f"{int(r * 100)}%\n{n:,}" for r, n in zip(RATIOS, N_TRAIN)]
    ))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("fraction of adaptation windows (log scale)", fontsize=10.5,
                  color=GREY_DARK, labelpad=10)
    ax.set_ylabel(r"45-min-horizon MSE  ($\times 10^{-3}$)", fontsize=10.5, color=GREY_DARK)

    # ---- direct right-edge labels (no legend box) -----------------------------
    labels = [
        ("Gate fusion", curves["gate"][-1], IKB, "bold"),
        ("Timer-XL · diff", curves["diff"][-1], INK, "normal"),
        ("Timer-XL · raw", curves["raw"][-1], GREY_DARK, "normal"),
        ("Time-MoE · zero-shot", curves["time_moe"][-1], GREY_MID, "normal"),
        ("Sundial · zero-shot", curves["sundial"][-1], GREY_MID, "normal"),
        ("STA-Conv-BiLSTM · diff", curves["STAConvBiLSTM_diff"][-1], GREY_MID, "normal"),
        ("TCN-Transformer · diff", curves["TCNTransformer_diff"][-1], GREY_MID, "normal"),
    ]
    ys = spread_labels([y for _, y, _, _ in labels], min_gap=0.85)
    for (text, y_true, color, weight), y_lab in zip(labels, ys):
        ax.text(1.03, y_lab, f"{text}   {y_true:.2f}",
                transform=ax.get_yaxis_transform(), fontsize=9.5, color=color,
                fontweight=weight, va="center", ha="left")
    group = [curves[f"{m}_diff"][-1] for m in ("CNNLSTM", "DiPCALSTM", "LSTMGRU")]
    ax.text(1.03, float(np.mean(group)),
            f"CNN/DiPCA/GRU · diff \n{min(group):.1f}–{max(group):.1f}",
            transform=ax.get_yaxis_transform(), fontsize=9.5, color=GREY_MID,
            va="center", ha="left")

    # ---- annotations ----------------------------------------------------------
    # -50% arrow at ratio 0.01 between gate and best DL baseline
    g0 = curves["gate"][0]
    b0 = curves["TCNTransformer_diff"][0]
    ax.annotate("", xy=(0.0115, g0), xytext=(0.0115, b0),
                arrowprops=dict(arrowstyle="<->", color=IKB, lw=1.4))
    ax.text(0.0122, (g0 + b0) / 2+0.25, "−50% vs best\nDL baseline", fontsize=10,
            color=IKB, va="center", fontweight="bold",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.6, alpha=0.85))

    ax.text(0.0105, 18.9, "FEW-SHOT REGIME · GATE BEST OF ALL 15 METHODS",
            fontsize=9, color=IKB, fontweight="bold", va="top",
            fontfamily="monospace")

    # honest footnote marker: STA-diff overtakes at >= 25% data
    sta = curves["STAConvBiLSTM_diff"]
    ax.scatter([0.25, 1.0], [sta[4], sta[6]], s=26, facecolor=PAPER,
               edgecolor=GREY_DARK, lw=1.1, zorder=5)
    ax.annotate("STA-diff overtakes\nat ≥25% data", xy=(0.25, sta[4]),
                xytext=(0.31, 4.6), fontsize=8.8, color=GREY_DARK,
                arrowprops=dict(arrowstyle="-", color=GREY_DARK, lw=0.8))

    # gate endpoint value tags
    ax.annotate(f"{curves['gate'][0]:.2f}", xy=(x[0], curves["gate"][0]),
                xytext=(0, -14), textcoords="offset points", ha="center",
                fontsize=10, color=IKB, fontweight="bold")
    ax.annotate(f"{curves['gate'][-1]:.2f}", xy=(x[-1], curves["gate"][-1]),
                xytext=(0, -14), textcoords="offset points", ha="center",
                fontsize=10, color=IKB, fontweight="bold")

    # ---- Swiss title block ----------------------------------------------------
    fig.subplots_adjust(left=0.075, right=0.775, top=0.925, bottom=0.185)
    # bar = plt.Rectangle((0.075, 0.955), 0.052, 0.011, transform=fig.transFigure,
    #                     facecolor=IKB, edgecolor="none")
    # fig.add_artist(bar)
    # fig.text(0.075, 0.895, "Gate fusion wins the few-shot regime",
    #          fontsize=19, color=INK, fontweight="bold", va="bottom")
    fig.text(0.075, 0.945,
             "TEP IDV13 · XMEAS10 purge rate",
             fontsize=10.5, color=GREY_DARK, va="bottom")
    fig.text(0.075, 0.014,
             "Experts: Timer-XL diff/raw (fine-tuned, 5-var input), Time-MoE 50M and Sundial 128M "
             "(zero-shot, constant across ratios)\n"
             "DL baselines: CNNLSTM, DiPCA-LSTM, LSTM-GRU, STA-Conv-BiLSTM, TCN-Transformer",
             fontsize=8.5, color=GREY_MID, va="bottom", linespacing=1.5)

    for stem in ("mse_vs_ratio",):
        fig.savefig(HERE / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(HERE / f"{stem}.pdf", facecolor=fig.get_facecolor())
    print(f"saved {HERE / 'mse_vs_ratio.png'} and .pdf")


if __name__ == "__main__":
    main()
