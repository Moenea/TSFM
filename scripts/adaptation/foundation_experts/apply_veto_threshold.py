"""Apply a fixed threshold to a trained Union-veto gate's probabilities."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def tv(a):
    return a[:, :, 0] if a.ndim == 3 else a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probabilities', type=Path, required=True)
    p.add_argument('--expert', action='append', required=True, help='diff then diff2')
    p.add_argument('--low', type=float, required=True)
    p.add_argument('--high', type=float, required=True)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    prob = np.load(a.probabilities)
    base = tv(np.load(Path(a.expert[0]) / 'pred.npy'))
    second = tv(np.load(Path(a.expert[1]) / 'pred.npy'))
    p1, p2 = base[:, :a.horizon], second[:, :a.horizon]
    hi, lo = np.maximum(p1, p2), np.minimum(p1, p2)
    union = p1.copy()
    union = np.where(hi > a.high, hi, union)
    union = np.where(lo < a.low, lo, union)
    out = base.copy()
    out[:, :a.horizon] = np.where((prob >= a.threshold)[:, None], union, p1)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(a.output_dir / 'pred.npy', out[:, :, None].astype(np.float32))
    np.save(a.output_dir / 'true.npy', tv(np.load(Path(a.expert[0]) / 'true.npy'))[:, :, None].astype(np.float32))
    np.save(a.output_dir / 'gate_probability.npy', prob.astype(np.float32))
    (a.output_dir / 'meta.json').write_text(
        '{"method":"Union-veto-gate-threshold","threshold":%.9g}\n' % a.threshold,
        encoding='utf-8')
    print(a.output_dir, 'threshold=', a.threshold)


if __name__ == '__main__':
    main()
