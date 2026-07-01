"""Aggregate per-ratio FM-gate metrics into a sample-count vs performance curve.

Mirrors scripts/adaptation/few_shot/TEP_IDV13/collect_curve.py but targets the
4-expert heterogeneous gate (diff, raw, time_moe, sundial) plus the fused gate output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERTS = ["diff", "raw", "time_moe", "sundial", "gate"]
METRICS = [
    "mse", "mae", "event_recall", "mean_lead_time_h",
    "pre_onset_false_alarm_rate", "window_precision", "window_recall",
]

# Full training-window count (used when log parsing is unavailable).
FULL_POOL = 12670


def tag_of(ratio: float) -> str:
    return "r" + repr(ratio).replace(".", "p")


def n_train_of(ratio: float, pool: int = FULL_POOL) -> int:
    return max(int(pool * ratio), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", type=Path, required=True)
    ap.add_argument("--ratios", nargs="+", type=float, required=True)
    ap.add_argument("--pool", type=int, default=FULL_POOL,
                    help="full training-window count (default 12670)")
    args = ap.parse_args()

    rows = []
    for ratio in sorted(args.ratios):
        tag = tag_of(ratio)
        mpath = args.summary_dir / f"metrics_{tag}.json"
        if not mpath.exists():
            print(f"WARN missing {mpath}; skipping ratio {ratio}")
            continue
        data = json.loads(mpath.read_text())
        n_train = n_train_of(ratio, args.pool)
        row: dict = {"ratio": ratio, "n_train": n_train}
        for exp in EXPERTS:
            md = data.get("models", {}).get(exp, {})
            for met in METRICS:
                row[f"{exp}_{met}"] = md.get(met)
        rows.append(row)

    # Write CSV
    cols = ["ratio", "n_train"] + [f"{e}_{k}" for e in EXPERTS for k in METRICS]
    csv_lines = [",".join(cols)]
    for row in rows:
        csv_lines.append(",".join(
            "" if row.get(c) is None else str(row.get(c)) for c in cols))
    csv_path = args.summary_dir / "curve.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Write JSON
    json_path = args.summary_dir / "curve.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Print table
    print(f"\n{'ratio':>6} {'n_train':>7} | {'diff_mse':>8} {'raw_mse':>8} "
          f"{'moe_mse':>8} {'gate_mse':>8} | "
          f"{'g_recall':>8} {'g_lead_h':>8} {'g_preFAR':>8}")
    print("-" * 85)
    for row in rows:
        lead = row.get("gate_mean_lead_time_h")
        diff_m = row.get("diff_mse")
        raw_m = row.get("raw_mse")
        moe_m = row.get("time_moe_mse")
        gate_m = row.get("gate_mse")
        recall = row.get("gate_event_recall")
        far = row.get("gate_pre_onset_false_alarm_rate")

        def fmt(v, w=8, prec=3):
            if v is None:
                return "-".rjust(w)
            return f"{v:.{prec}f}".rjust(w)

        print(f"{row['ratio']:>6} {row['n_train']:>7} | "
              f"{fmt(diff_m)} {fmt(raw_m)} {fmt(moe_m)} {fmt(gate_m)} | "
              f"{fmt(recall)} {fmt(lead)} {fmt(far, prec=4)}")

    print(f"\nCSV  -> {csv_path}")
    print(f"JSON -> {json_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def xy(key):
            pts = [(r["n_train"], r[key]) for r in rows if r.get(key) is not None]
            return [p[0] for p in pts], [p[1] for p in pts]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Panel 0: MSE vs # train windows per expert + gate
        markers = {"diff": ("s-", "tab:blue"), "raw": ("o-", "tab:orange"),
                   "time_moe": ("D-", "tab:purple"), "sundial": ("v-", "tab:brown"),
                   "gate": ("^-", "tab:green")}
        for exp, (mk, col) in markers.items():
            x, y = xy(f"{exp}_mse")
            if x:
                axes[0].plot(x, y, mk, color=col, label=exp, linewidth=1.5, markersize=5)
        axes[0].set(xscale="log", xlabel="# train windows", ylabel="MSE (h=15)",
                    title="Forecast MSE vs few-shot size")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Panel 1: gate mean lead time
        x, y = xy("gate_mean_lead_time_h")
        axes[1].plot(x, y, "^-", color="tab:green", linewidth=1.5, markersize=5)
        axes[1].set(xscale="log", xlabel="# train windows", ylabel="lead time (h)",
                    title="Gate mean lead time")
        axes[1].grid(True, alpha=0.3)

        # Panel 2: gate pre-onset FAR
        x, y = xy("gate_pre_onset_false_alarm_rate")
        axes[2].plot(x, y, "^-", color="tab:red", linewidth=1.5, markersize=5,
                     label="gate")
        # also overlay diff FAR for reference
        x2, y2 = xy("diff_pre_onset_false_alarm_rate")
        if x2:
            axes[2].plot(x2, y2, "s--", color="tab:blue", linewidth=1.2,
                         markersize=4, label="diff")
        axes[2].set(xscale="log", xlabel="# train windows", ylabel="pre-onset FAR",
                    title="Pre-onset false-alarm rate")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()

        # Save to summary dir
        png1 = args.summary_dir / "curve.png"
        fig.savefig(png1, dpi=120)
        print(f"plot -> {png1}")

        # Also save to figures dir
        import os
        figures_dir = Path("/home/aicode/sherwin/TSFM/figures/few-shot/TEP/IDV13")
        figures_dir.mkdir(parents=True, exist_ok=True)
        png2 = figures_dir / "fm_curve.png"
        fig.savefig(png2, dpi=120)
        print(f"plot -> {png2}")

        plt.close(fig)
    except ImportError as exc:
        print(f"(plot skipped: matplotlib not available — {exc})")


if __name__ == "__main__":
    main()
