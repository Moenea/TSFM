"""Create raw-aligned first differences for the ten exported TEP IDV13 runs."""
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("/home/aicode/sherwin/dataset/TEP/csv")
DESTINATION = Path("/home/aicode/sherwin/dataset/TEP/csv_diff")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE.glob("*IDV13*Run*.csv"))
    if len(paths) != 10:
        raise RuntimeError(f"expected 10 IDV13 CSV files, found {len(paths)}")

    for path in paths:
        frame = pd.read_csv(path)
        if frame.columns[0] != "Time":
            raise ValueError(f"expected leading Time column in {path}")
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {path}")
        differences = np.zeros_like(values)
        differences[1:] = np.diff(values, axis=0)
        output = frame.copy()
        output.iloc[:, 1:] = differences
        output.to_csv(DESTINATION / path.name, index=False)
        print(f"wrote {DESTINATION / path.name} shape={output.shape}")


if __name__ == "__main__":
    main()
