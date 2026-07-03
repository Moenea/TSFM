"""Aggregate XMEAS10 per-ratio metrics into a comparison table + figure:
5 DL baselines (MS, 5-var, BOTH raw & diff) vs the TSFM experts vs the Gate."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aicode/sherwin/TSFM")
SUMMARY = ROOT / "results/TEP_IDV13_XMEAS10_Summary"
FIGDIR = ROOT / "figures/few-shot/TEP/IDV13_XMEAS10"
RATIOS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
POOL = 12670
BASE_MODELS = ["CNNLSTM", "DiPCALSTM", "LSTMGRU", "STAConvBiLSTM", "TCNTransformer"]
MODES = ["raw", "diff"]
BASELINES = [f"{m}_{md}" for m in BASE_MODELS for md in MODES]  # 10
EXPERTS = ["diff", "raw", "time_moe", "sundial"]
ORDER = BASELINES + EXPERTS + ["gate"]
COLOR = {"CNNLSTM": "tab:blue", "DiPCALSTM": "tab:orange", "LSTMGRU": "tab:green",
         "STAConvBiLSTM": "tab:red", "TCNTransformer": "tab:purple"}


def tag(r): return "r" + repr(r).replace(".", "p")


def main() -> None:
    rows = {}
    for r in RATIOS:
        f = SUMMARY / f"metrics_{tag(r)}.json"
        if f.exists():
            rows[r] = json.loads(f.read_text())["models"]
        else:
            print(f"WARN missing {f}; skipping ratio {r}")
    if not rows:
        raise SystemExit("no metrics_*.json found; run run_gate_eval.sh per ratio first")

    cols = ["ratio", "n_train"] + [f"{m}_mse" for m in ORDER] + [f"{m}_recall" for m in ORDER]
    lines = [",".join(cols)]
    for r in RATIOS:
        if r not in rows:
            continue
        md = rows[r]
        rec = [r, max(int(POOL * r), 1)]
        rec += [round(md.get(m, {}).get("mse", float("nan")), 2) for m in ORDER]
        rec += [md.get(m, {}).get("event_recall", float("nan")) for m in ORDER]
        lines.append(",".join(map(str, rec)))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "comparison.csv").write_text("\n".join(lines) + "\n")

    xs = [r for r in RATIOS if r in rows]
    nt = [max(int(POOL * r), 1) for r in xs]

    def series(key, metric):
        return [rows[r].get(key, {}).get(metric, float("nan")) for r in xs]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for m in BASE_MODELS:
        ax[0].plot(nt, series(f"{m}_diff", "mse"), "-", color=COLOR[m], lw=1.6, marker="o", ms=4, label=f"{m} (diff)")
        ax[0].plot(nt, series(f"{m}_raw", "mse"), "--", color=COLOR[m], lw=1.0, marker="x", ms=4, alpha=0.6, label=f"{m} (raw)")
    ax[0].plot(nt, series("diff", "mse"), ":", color="gray", lw=1.2, label="Timer-XL diff")
    ax[0].plot(nt, series("gate", "mse"), "^-", color="black", lw=2.6, ms=8, label="Gate (ours)")
    ax[0].set(xscale="log", xlabel="# train windows", ylabel="XMEAS10 MSE (h=15)",
              title="XMEAS10 (Purge flow): DL baselines (raw & diff) vs Gate")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=0.3)

    for m in BASE_MODELS:
        ax[1].plot(nt, series(f"{m}_diff", "event_recall"), "-", color=COLOR[m], lw=1.4, marker="o", ms=4, label=f"{m} (diff)")
    ax[1].plot(nt, series("gate", "event_recall"), "^-", color="black", lw=2.6, ms=8, label="Gate (ours)")
    ax[1].set(xscale="log", xlabel="# train windows", ylabel="event recall",
              title="Fault detection (diff baselines) vs few-shot size", ylim=(-0.05, 1.08))
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "baselines_vs_gate.png", dpi=140)
    fig.savefig(SUMMARY / "comparison.png", dpi=140)
    print(f"wrote {SUMMARY/'comparison.csv'} + {FIGDIR/'baselines_vs_gate.png'}")


if __name__ == "__main__":
    main()
