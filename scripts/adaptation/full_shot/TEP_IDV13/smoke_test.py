"""Static checks for TEP split files, windows, and raw-aligned differences."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/aicode/sherwin/TSFM")
from data_provider.data_loader import MultivariateDatasetYAMLSplit


ROOT = Path("/home/aicode/sherwin/dataset/TEP")
SETTING = Path("/home/aicode/sherwin/TSFM/setting")
TARGET = "XMEAS07 Reactor Pressure"
SIZE = [96, 96, 96]


def check_dataset(split: str, flag: str) -> None:
    dataset = MultivariateDatasetYAMLSplit(
        root_path=str(ROOT),
        flag=flag,
        size=SIZE,
        split_file=str(SETTING / split),
        features="S",
        target=TARGET,
    )
    x, y, x_mark, y_mark = dataset[0]
    assert x.shape == (96, 1), x.shape
    assert y.shape == (96, 1), y.shape
    assert np.isfinite(np.asarray(x)).all()
    assert np.isfinite(np.asarray(y)).all()
    print(split, flag, len(dataset), tuple(x.shape), tuple(y.shape))


def main() -> None:
    for split in ("TEP_IDV13_XMEAS07.yaml", "TEP_IDV13_XMEAS07_diff.yaml"):
        for flag in ("train", "val", "test"):
            check_dataset(split, flag)

    raw_path = next((ROOT / "csv").glob("*Run1.csv"))
    diff_path = ROOT / "csv_diff" / raw_path.name
    raw = pd.read_csv(raw_path)
    diff = pd.read_csv(diff_path)
    assert np.allclose(raw["Time"], diff["Time"])
    expected = np.diff(raw[TARGET].to_numpy())
    assert diff[TARGET].iloc[0] == 0.0
    assert np.allclose(diff[TARGET].iloc[1:].to_numpy(), expected)
    print("difference alignment OK")


if __name__ == "__main__":
    main()
