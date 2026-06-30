# scripts/adaptation/foundation_experts/tests/test_smoke_time_moe.py
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path("/home/aicode/sherwin/TSFM")
FM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"  # update from PROBE.md if isolated env
ADAPTER = ROOT / "scripts/adaptation/foundation_experts/time_moe/adapter.py"
VAL_SPLIT = ROOT / "setting/TEP_IDV13_XMEAS07_val.yaml"   # test=Run8
DATA_ROOT = "/home/aicode/sherwin/dataset/TEP"
TARGET = "XMEAS07 Reactor Pressure"


def test_zero_shot_predict_shape():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tm"
        r = subprocess.run([FM_PY, str(ADAPTER), "--mode", "predict", "--zero-shot",
            "--split-file", str(VAL_SPLIT), "--data-root", DATA_ROOT, "--target", TARGET,
            "--horizon", "15", "--out-dir", str(out), "--device", "cuda:0"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        pred = np.load(out / "pred.npy")
        true = np.load(out / "true.npy")
        assert pred.shape == (1810, 15, 1), pred.shape   # Run8 windows
        assert true.shape == (1810, 15, 1), true.shape
        assert np.isfinite(pred).all()
        # sanity: original-scale reactor pressure is O(1e3), not normalized O(1)
        assert pred.mean() > 100.0, pred.mean()


if __name__ == "__main__":
    test_zero_shot_predict_shape()
    print("ALL PASS")
