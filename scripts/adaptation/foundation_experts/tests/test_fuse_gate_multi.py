# scripts/adaptation/foundation_experts/tests/test_fuse_gate_multi.py
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
import fuse_gate_multi as g  # noqa: E402


def test_weights_are_a_simplex():
    gate = g.GateMLPMulti(in_dim=8, hidden=16, horizon=15, n_experts=5)
    feats = torch.randn(20, 8)
    w = gate(feats)
    assert w.shape == (20, 15, 5)
    sums = w.sum(dim=-1).detach().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert (w.detach().numpy() >= 0).all()


def test_fuse_is_weighted_sum():
    # 2 experts, horizon 3, batch 2
    stack = np.array([
        [[1., 1., 1.], [2., 2., 2.]],   # expert 0
        [[3., 3., 3.], [4., 4., 4.]],   # expert 1
    ])  # (N=2, B=2, H=3)
    weights = np.zeros((2, 3, 2))
    weights[..., 0] = 0.25
    weights[..., 1] = 0.75
    fused = g.fuse(weights, stack)
    assert fused.shape == (2, 3)
    # window0: 0.25*1 + 0.75*3 = 2.5 ; window1: 0.25*2 + 0.75*4 = 3.5
    assert np.allclose(fused[0], 2.5) and np.allclose(fused[1], 3.5)


if __name__ == "__main__":
    test_weights_are_a_simplex()
    test_fuse_is_weighted_sum()
    print("ALL PASS")
