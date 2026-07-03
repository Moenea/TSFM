"""Build the 5-variable MS datasets for the XMEAS10 (Purge flow) study.

Creates (idempotent):
  csv_5var/       : [XMEAS15, XMEAS17, XMV06, XMV11, XMEAS10]  (target XMEAS10 LAST)
  csv_5var_diff/  : first-difference of csv_5var (row 0 = 0), for diff experts/baselines

The 5 variables (Purge flow + stripper level/underflow + purge valve + separator
condenser-coolant valve) are the physically-relevant covariates for Purge flow.
Target = XMEAS10 last so the MS --covariate --last_token convention selects it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("/home/aicode/sherwin/dataset/TEP/csv")
V5 = Path("/home/aicode/sherwin/dataset/TEP/csv_5var")
V5D = Path("/home/aicode/sherwin/dataset/TEP/csv_5var_diff")
COVS = ["XMEAS15 Stripper Level", "XMEAS17 Stripper Underflow",
        "XMV06 Purge", "XMV11 Condenser Coolant"]
TARGET = "XMEAS10 Purge Rate"
COLS = COVS + [TARGET]


def main() -> None:
    V5.mkdir(parents=True, exist_ok=True)
    V5D.mkdir(parents=True, exist_ok=True)
    paths = sorted(SRC.glob("*IDV13*Run*.csv"))
    if len(paths) != 10:
        raise RuntimeError(f"expected 10 IDV13 CSVs, found {len(paths)}")
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns {missing} in {p}")
        sub = df[COLS]
        sub.to_csv(V5 / p.name, index=False)
        vals = sub.to_numpy(dtype=np.float64)
        d = np.zeros_like(vals)
        d[1:] = np.diff(vals, axis=0)
        pd.DataFrame(d, columns=COLS).to_csv(V5D / p.name, index=False)
    print(f"wrote {len(paths)} csv_5var + {len(paths)} csv_5var_diff (cols: {COLS})")


if __name__ == "__main__":
    main()
