"""Aggregate per-ratio few-shot metrics into a sample-count vs performance curve."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODELS = ["raw", "diff", "gate_t2"]
METRICS = [
    "mse", "mae", "event_recall", "mean_lead_time_h",
    "pre_onset_false_alarm_rate", "window_precision", "window_recall",
]


def tag_of(ratio: float) -> str:
    return "r" + repr(ratio).replace(".", "p")


def parse_n_train(log_path: Path):
    """First `train <N>` line printed by data_provider for this ratio."""
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.match(r"^train (\d+)\s*$", line.strip())
        if m:
            return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--ratios", nargs="+", type=float, required=True)
    ap.add_argument("--pool", type=int, default=12670,
                    help="full train window count, used only if log parsing fails")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for ratio in sorted(args.ratios):
        tag = tag_of(ratio)
        mpath = args.summary_dir / f"metrics_{tag}.json"
        if not mpath.exists():
            print(f"WARN missing {mpath}; skipping ratio {ratio}")
            continue
        data = json.loads(mpath.read_text())
        n_train = parse_n_train(args.log_dir / f"few_{tag}.log")
        if n_train is None:
            n_train = max(int(args.pool * ratio), 1)
        row = {"ratio": ratio, "n_train": n_train}
        for mdl in MODELS:
            md = data["models"][mdl]
            for met in METRICS:
                row[f"{mdl}_{met}"] = md.get(met)
        rows.append(row)

    cols = ["ratio", "n_train"] + [f"{m}_{k}" for m in MODELS for k in METRICS]
    csv_lines = [",".join(cols)]
    for row in rows:
        csv_lines.append(",".join(
            "" if row.get(c) is None else str(row.get(c)) for c in cols))
    args.out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'ratio':>6} {'n_train':>7} | {'raw_mse':>8} {'diff_mse':>8} "
          f"{'gate_mse':>8} | {'g_recall':>8} {'g_lead_h':>8} {'g_preFAR':>8}")
    for row in rows:
        lead = row["gate_t2_mean_lead_time_h"]
        print(f"{row['ratio']:>6} {row['n_train']:>7} | "
              f"{row['raw_mse']:>8.2f} {row['diff_mse']:>8.2f} {row['gate_t2_mse']:>8.2f} | "
              f"{row['gate_t2_event_recall']:>8.2f} "
              f"{('-' if lead is None else f'{lead:.3f}'):>8} "
              f"{row['gate_t2_pre_onset_false_alarm_rate']:>8.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def xy(key):
            pts = [(r["n_train"], r[key]) for r in rows if r[key] is not None]
            return [p[0] for p in pts], [p[1] for p in pts]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for key, mk, lab in [("raw_mse", "o-", "raw"),
                             ("diff_mse", "s-", "diff"),
                             ("gate_t2_mse", "^-", "gate_t2")]:
            x, y = xy(key)
            ax[0].plot(x, y, mk, label=lab)
        ax[0].set(xscale="log", xlabel="# train windows", ylabel="MSE (h=15)",
                  title="Forecast MSE vs few-shot size")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)

        x, y = xy("gate_t2_mean_lead_time_h")
        ax[1].plot(x, y, "^-", color="tab:green")
        ax[1].set(xscale="log", xlabel="# train windows", ylabel="lead time (h)",
                  title="Gate-T2 mean lead time")
        ax[1].grid(True, alpha=0.3)

        for key, mk, c, lab in [("gate_t2_pre_onset_false_alarm_rate", "^-", "tab:red", "gate_t2"),
                                ("diff_pre_onset_false_alarm_rate", "s--", "tab:orange", "diff")]:
            x, y = xy(key)
            ax[2].plot(x, y, mk, color=c, label=lab)
        ax[2].set(xscale="log", xlabel="# train windows", ylabel="pre-onset FAR",
                  title="Pre-onset false-alarm rate")
        ax[2].legend(); ax[2].grid(True, alpha=0.3)

        fig.tight_layout()
        out_png = args.summary_dir / "curve.png"
        fig.savefig(out_png, dpi=120)
        print(f"\nplot -> {out_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
