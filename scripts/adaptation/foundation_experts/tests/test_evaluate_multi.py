# scripts/adaptation/foundation_experts/tests/test_evaluate_multi.py
import sys, tempfile, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/aicode/sherwin/TSFM")
TARGET = "XMEAS07 Reactor Pressure"
TSFM_PY = "/home/aicode/miniconda3/envs/tsfm/bin/python"


def test_report_has_each_expert():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "csv").mkdir()
        n = 300
        t = np.arange(n) * 0.05
        df = pd.DataFrame({"Time": t, TARGET: 2700.0 + np.zeros(n)})
        df.to_csv(tmp / "csv/Run.csv", index=False)
        split = tmp / "split.yaml"
        split.write_text(f'target: "{TARGET}"\ntrain:\n  - csv/Run.csv\ntest:\n  - csv/Run.csv\n')
        nwin = 300 - 96 - 96 + 1
        for name in ("a", "b"):
            dd = tmp / name; dd.mkdir()
            np.save(dd / "pred.npy", np.full((nwin, 15, 1), 2700.0, np.float32))
            np.save(dd / "true.npy", np.full((nwin, 15, 1), 2700.0, np.float32))
        out = tmp / "report.json"
        r = subprocess.run([TSFM_PY,
            str(ROOT / "scripts/adaptation/foundation_experts/evaluate_multi.py"),
            "--data-root", str(tmp), "--split", str(split), "--target", TARGET,
            "--expert", f"a:{tmp/'a'}", "--expert", f"b:{tmp/'b'}",
            "--output", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.loads(out.read_text())
        assert set(rep["models"]) == {"a", "b"}
        assert rep["models"]["a"]["mse"] == 0.0


if __name__ == "__main__":
    test_report_has_each_expert()
    print("ALL PASS")
