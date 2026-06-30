# scripts/adaptation/foundation_experts/tests/test_expert_io.py
import sys, tempfile, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
from common import expert_io as io  # noqa: E402

TARGET = "XMEAS07 Reactor Pressure"


def _make_split(tmp: Path, n_rows: int):
    """One synthetic 'test' file with a ramping target so windows are distinguishable."""
    csv_dir = tmp / "csv"; csv_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_rows) * 0.05
    df = pd.DataFrame({"Time": t, TARGET: 1000.0 + np.arange(n_rows, dtype=float)})
    df.to_csv(csv_dir / "Run.csv", index=False)
    split = tmp / "split.yaml"
    split.write_text(
        f'target: "{TARGET}"\ntrain:\n  - csv/Run.csv\nval:\n  - csv/Run.csv\ntest:\n  - csv/Run.csv\n'
    )
    return split


def test_usable_count():
    assert io.usable_count(2001, 96, 96) == 1810


def test_iter_infer_windows_alignment():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        split = _make_split(tmp, n_rows=200)
        ctx, true = io.iter_infer_windows(tmp, split, TARGET, seq_len=96, pred_len=96, horizon=15)
        # count uses pred_len=96 -> 200-96-96+1 = 9 windows
        assert ctx.shape == (9, 96), ctx.shape
        assert true.shape == (9, 15), true.shape
        # window 0: context = values[0:96] = 1000..1095 ; true = values[96:111] = 1096..1110
        assert ctx[0, 0] == 1000.0 and ctx[0, -1] == 1095.0
        assert true[0, 0] == 1096.0 and true[0, -1] == 1110.0
        # window 1 shifted by exactly one step
        assert ctx[1, 0] == 1001.0 and true[1, 0] == 1097.0


def test_save_result_shape_and_units():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "expert"
        pred = np.ones((5, 15), dtype=np.float32) * 2700.0
        true = np.ones((5, 15), dtype=np.float32) * 2710.0
        io.save_result(out, pred, true, {"model": "unit-test"})
        p = np.load(out / "pred.npy"); t = np.load(out / "true.npy")
        assert p.shape == (5, 15, 1) and t.shape == (5, 15, 1)
        assert p.dtype == np.float32 and abs(float(p.mean()) - 2700.0) < 1e-3
        assert json.loads((out / "meta.json").read_text())["model"] == "unit-test"


if __name__ == "__main__":
    test_usable_count()
    test_iter_infer_windows_alignment()
    test_save_result_shape_and_units()
    print("ALL PASS")
