"""Aggregate the per-ratio batch_metrics summaries of the XMEAS10 few-shot
study into one cross-ratio table and a 3-panel curve figure.

Reads  results/"XMEAS10 Purge Rate_Summary_{ratio}"/summary.csv (7 ratios)
Writes results/TEP_IDV13_XMEAS10_Summary/batch_metrics_by_ratio.csv
       results/TEP_IDV13_XMEAS10_Summary/batch_metrics_{metric}_pivot.csv
       figures/few-shot/TEP/IDV13_XMEAS10/batch_metrics_curves.png/.pdf

Plotted metrics are the `_clean` variants (context of 30 true samples before
the forecast origin contains no alarm), i.e. genuine advance prognosis.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TSFM = Path("/home/aicode/sherwin/TSFM")
RESULTS = TSFM / "results"
OUT_DIR = RESULTS / "TEP_IDV13_XMEAS10_Summary"
FIG_DIR = TSFM / "figures" / "few-shot" / "TEP" / "IDV13_XMEAS10"
RATIOS = ["r0p01", "r0p02", "r0p05", "r0p1", "r0p25", "r0p5", "r1p0"]
N_TRAIN = {"r0p01": 126, "r0p02": 253, "r0p05": 633, "r0p1": 1267,
           "r0p25": 3167, "r0p5": 6335, "r1p0": 12670}
STEP_MIN = 3.0  # one prediction step = 3 min

KEEP_COLS = [
    "ratio_pred_in_true_alarm_patches", "ratio_pred_in_no_true_alarm_patches",
    "mean_lead_time_patch", "mean_prognosis_error",
    "ratio_pred_in_true_alarm_patches_clean", "ratio_pred_in_no_true_alarm_patches_clean",
    "mean_lead_time_patch_clean", "mean_prognosis_error_clean",
    "ratio_pred_in_true_alarm_patches_qf", "ratio_pred_in_no_true_alarm_patches_qf",
    "mean_lead_time_patch_qf",
    "ratio_pred_in_true_alarm_patches_clean_qf", "ratio_pred_in_no_true_alarm_patches_clean_qf",
    "mean_lead_time_patch_clean_qf",
    "n_pred_alarm_patches", "n_quality_rejected",
    "tp_window_clean", "fp_window_clean", "fn_window_clean", "tn_window_clean",
]

PIVOT_METRICS = {
    "recall_clean": "ratio_pred_in_true_alarm_patches_clean",
    "false_rate_clean": "ratio_pred_in_no_true_alarm_patches_clean",
    "lead_time_clean": "mean_lead_time_patch_clean",
}


def load_all() -> pd.DataFrame:
    rows = []
    for r in RATIOS:
        csv = RESULTS / f"XMEAS10 Purge Rate_Summary_{r}" / "summary.csv"
        if not csv.exists():
            print(f"missing {csv} — skipped")
            continue
        df = pd.read_csv(csv)
        df.insert(0, "ratio", r)
        df.insert(1, "n_train", N_TRAIN[r])
        rows.append(df[["ratio", "n_train", "model"] + KEEP_COLS])
    if not rows:
        raise SystemExit("no per-ratio summaries found — run batch_metrics first")
    return pd.concat(rows, ignore_index=True)


def plot_curves(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panels = [
        ("ratio_pred_in_true_alarm_patches_clean", "Recall rate (clean)", 1.0),
        ("ratio_pred_in_no_true_alarm_patches_clean", "False prognosis rate (clean)", 1.0),
        ("mean_lead_time_patch_clean", "Mean lead time (clean, min)", STEP_MIN),
    ]
    models = list(long["model"].drop_duplicates())
    dl = [m for m in models if m not in
          ("Gate", "Timer-XL-diff", "Timer-XL-raw", "Time-MoE-ZS", "Sundial-ZS")]
    style = {m: dict(color="#b8b8b8", lw=1.0, marker="o", ms=2.5, zorder=1)
             for m in dl}
    style.update({
        "Time-MoE-ZS": dict(color="#8a94a6", lw=1.3, ls="--", marker="s", ms=3.5, zorder=2),
        "Sundial-ZS": dict(color="#8a94a6", lw=1.3, ls=":", marker="D", ms=3.5, zorder=2),
        "Timer-XL-raw": dict(color="#555555", lw=1.6, marker="o", ms=4, zorder=3),
        "Timer-XL-diff": dict(color="#c04a3a", lw=1.8, marker="o", ms=4, zorder=4),
        "Gate": dict(color="#002FA7", lw=2.4, marker="^", ms=6, zorder=5),
    })
    for ax, (col, title, scale) in zip(axes, panels):
        dl_labelled = False
        for m in models:
            sub = long[long["model"] == m].sort_values("n_train")
            kw = dict(style[m])
            if m in dl:
                label = None if dl_labelled else "DL baselines (10)"
                dl_labelled = True
            else:
                label = m
            ax.plot(sub["n_train"], sub[col] * scale, label=label, **kw)
        ax.set(xscale="log", xlabel="# train windows", title=title)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("XMEAS10 prognosis metrics vs few-shot size  "
                 "(window-level, $\\mu\\pm3\\sigma$ limits)", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"batch_metrics_curves.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'batch_metrics_curves.png'}")


def main() -> None:
    long = load_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "batch_metrics_by_ratio.csv"
    long.to_csv(out_csv, index=False)
    print(f"saved {out_csv}  ({len(long)} rows)")
    for tag, col in PIVOT_METRICS.items():
        piv = long.pivot(index="model", columns="ratio", values=col)[RATIOS]
        piv.to_csv(OUT_DIR / f"batch_metrics_{tag}_pivot.csv")
    plot_curves(long)


if __name__ == "__main__":
    main()
