"""Recall vs false-prognosis-rate operating points (clean windows) for the
XMEAS10 few-shot study, with an industrial false-alarm budget (FPR <= 5%,
EEMUA-191 / ISA-18.2 spirit).

Reads  results/TEP_IDV13_XMEAS10_Summary/batch_metrics_by_ratio.csv
Writes figures/few-shot/TEP/IDV13_XMEAS10/pareto_clean.png/.pdf
       results/TEP_IDV13_XMEAS10_Summary/recall_at_far5_by_ratio.csv

Headline claim it substantiates: among methods that stay inside the 5%
false-alarm budget at EVERY few-shot size, Gate has the highest recall at
every size; every DL baseline violates the budget at one size or another.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TSFM = Path("/home/aicode/sherwin/TSFM")
SUMMARY = TSFM / "results" / "TEP_IDV13_XMEAS10_Summary"
FIG_DIR = TSFM / "figures" / "few-shot" / "TEP" / "IDV13_XMEAS10"
RATIOS = ["r0p01", "r0p02", "r0p05", "r0p1", "r0p25", "r0p5", "r1p0"]
N_TRAIN = {"r0p01": 126, "r0p02": 253, "r0p05": 633, "r0p1": 1267,
           "r0p25": 3167, "r0p5": 6335, "r1p0": 12670}
BUDGET = 0.05

STYLE = {
    "Time-MoE-ZS": dict(color="#8a94a6", marker="s"),
    "Sundial-ZS": dict(color="#8a94a6", marker="D"),
    "Timer-XL-raw": dict(color="#555555", marker="o"),
    "Timer-XL-diff": dict(color="#c04a3a", marker="o"),
    "Gate": dict(color="#002FA7", marker="^"),
}
DL_COLOR = "#b8b8b8"


def main() -> None:
    df = pd.read_csv(SUMMARY / "batch_metrics_by_ratio.csv")
    df["rec"] = df["ratio_pred_in_true_alarm_patches_clean"]
    df["far"] = df["ratio_pred_in_no_true_alarm_patches_clean"]
    models = list(df["model"].drop_duplicates())
    dl = [m for m in models if m not in STYLE]

    # budget compliance at every ratio
    compliant = sorted(
        m for m in models
        if (df.loc[df.model == m, "far"] <= BUDGET).all()
    )
    print("always inside 5% budget:", compliant)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # --- Panel A: operating points, 7 ratios per model ---
    for m in models:
        sub = df[df.model == m]
        kw = STYLE.get(m, dict(color=DL_COLOR, marker="o"))
        big = m in STYLE
        axA.scatter(sub["far"], sub["rec"], s=34 if big else 20,
                    color=kw["color"], marker=kw["marker"],
                    alpha=0.9 if big else 0.55,
                    edgecolors="white", linewidths=0.4,
                    zorder=5 if m == "Gate" else (4 if big else 2),
                    label=m if big else None)
    axA.scatter([], [], s=20, color=DL_COLOR, marker="o", alpha=0.55,
                label="DL baselines (10)")
    axA.axvspan(0, BUDGET, color="#2e8b57", alpha=0.10, zorder=1)
    axA.axvline(BUDGET, color="#2e8b57", lw=1.0, ls="--")
    axA.annotate("false-alarm budget\nFPR $\\leq$ 5%", xy=(BUDGET, 0.93),
                 xytext=(0.09, 0.88), fontsize=8, color="#2e8b57",
                 arrowprops=dict(arrowstyle="-", color="#2e8b57", lw=0.8))
    axA.set(xlabel="False prognosis rate (clean windows)",
            ylabel="Recall rate (clean windows)",
            title="Operating points, all 7 few-shot sizes per method",
            xlim=(-0.02, 1.0), ylim=(-0.03, 1.02))
    axA.grid(alpha=0.3, lw=0.5)
    axA.legend(fontsize=7.5, frameon=False, loc="upper right")

    # --- Panel B: recall vs n_train, budget-compliant methods only ---
    for m in models:
        sub = df[df.model == m].sort_values("n_train")
        if m in compliant:
            kw = STYLE.get(m, dict(color=DL_COLOR, marker="o"))
            axB.plot(sub["n_train"], sub["rec"], color=kw["color"],
                     marker=kw["marker"], ms=6 if m == "Gate" else 4,
                     lw=2.2 if m == "Gate" else 1.3,
                     zorder=5 if m == "Gate" else 3, label=m)
        else:
            axB.plot(sub["n_train"], sub["rec"], color=DL_COLOR, lw=0.8,
                     alpha=0.35, zorder=1)
    axB.plot([], [], color=DL_COLOR, lw=0.8, alpha=0.5,
             label="budget violated\n(shown faded)")
    axB.set(xscale="log", xlabel="# train windows",
            ylabel="Recall rate (clean windows)",
            title=f"Recall of methods inside the {BUDGET:.0%} budget "
                  "at every size")
    axB.grid(alpha=0.3, lw=0.5)
    axB.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle("XMEAS10 advance prognosis under a false-alarm budget  "
                 "(window-level, clean context, $\\mu\\pm3\\sigma$)", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"pareto_clean.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'pareto_clean.png'}")

    # --- table: recall by ratio with compliance flags ---
    rec = df.pivot(index="model", columns="ratio", values="rec")[RATIOS]
    far = df.pivot(index="model", columns="ratio", values="far")[RATIOS]
    out = rec.round(3).astype(str)
    out[far > BUDGET] = out[far > BUDGET] + "*"
    out["always_in_budget"] = [m in compliant for m in out.index]
    out["far_max"] = far.max(axis=1).round(3)
    out = out.sort_values(["always_in_budget", RATIOS[-1]], ascending=False)
    csv = SUMMARY / "recall_at_far5_by_ratio.csv"
    out.to_csv(csv)
    print(f"saved {csv}  (* = false rate above {BUDGET:.0%} at that size)")
    print(out.to_string())


if __name__ == "__main__":
    main()
