"""3-way clean-window comparison: exp A / exp B / alarm gate, on Run9-10 and the
extended test (Run18-27), all under mag=25 limits. Reads the batch_metrics summaries
and prints recall/FAR/lead + combined score S; checks the acceptance criterion on
the (reliable, 10-event) extended test."""
from pathlib import Path
import pandas as pd

ROOT = Path("/home/aicode/sherwin/TSFM")
ORDER = ["Gate-expA", "Gate-expB", "Gate-alarm"]
LABEL = {"Gate-expA": "exp A (mag100 experts, MSE gate)",
         "Gate-expB": "exp B (mag25 experts, MSE gate)",
         "Gate-alarm": "alarm gate (mag25, alarm-aware)"}


def load(summary):
    d = pd.read_csv(summary)
    d = d[d.model.isin(ORDER)].copy()
    d["recall"] = d.ratio_pred_in_true_alarm_patches_clean
    d["FAR"] = d.ratio_pred_in_no_true_alarm_patches_clean
    d["lead_min"] = d.mean_lead_time_patch_clean * 3
    d["S"] = d.recall + (1 - d.FAR / 0.05) + d.mean_lead_time_patch_clean / 15
    d["method"] = pd.Categorical(d.model, categories=ORDER, ordered=True)
    return d.sort_values("method")[["model", "recall", "FAR", "lead_min", "S"]]


def show(tag, summary):
    d = load(summary)
    print(f"\n===== {tag} (mag=25 limits) =====")
    print(d.assign(label=d.model.map(LABEL))[["label", "recall", "FAR", "lead_min", "S"]]
          .to_string(index=False,
                     formatters={"recall": "{:.3f}".format, "FAR": "{:.3f}".format,
                                 "lead_min": "{:.2f}".format, "S": "{:.3f}".format}))
    return d.set_index("model")


def main():
    show("Run9-10 (comparable to prior)", ROOT / "results/XMEAS10 Purge Rate_Summary_final910/summary.csv")
    ext = show("Extended test Run18-27 (10 events, reliable)",
               ROOT / "results/XMEAS10 Purge Rate_Summary_finalext/summary.csv")

    a, g = ext.loc["Gate-expA"], ext.loc["Gate-alarm"]
    print("\n===== ACCEPTANCE (extended test, alarm gate vs exp A) =====")
    dom_r = g.recall >= a.recall
    dom_f = g.FAR <= a.FAR
    dom_l = g.lead_min >= a.lead_min
    print(f"recall  {g.recall:.3f} vs {a.recall:.3f}  {'PASS' if dom_r else 'FAIL'}")
    print(f"FAR     {g.FAR:.3f} vs {a.FAR:.3f}  {'PASS' if dom_f else 'FAIL'}")
    print(f"lead    {g.lead_min:.2f} vs {a.lead_min:.2f} min  {'PASS' if dom_l else 'FAIL'}")
    print(f"combined S  {g.S:.3f} vs {a.S:.3f}  {'PASS' if g.S >= a.S else 'FAIL'}")
    print("DOMINATES ALL THREE:" , bool(dom_r and dom_f and dom_l))


if __name__ == "__main__":
    main()
