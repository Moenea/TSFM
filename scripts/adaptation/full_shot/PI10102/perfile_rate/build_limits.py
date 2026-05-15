#!/usr/bin/env python
"""
Per-file Δ-space alarm-limit builder for PI10102 test transitions.

For every test transition file listed in a batch_metrics yaml config, compute
the absolute first-difference distribution of the target column and set the
alarm threshold τ_file = quantile(|Δx|, q). Limits are two-sided in Δ space:

    HH = +τ_file,  H = +τ_file,  L = -τ_file,  LL = -τ_file

The resulting CSV has one row per test file and is consumed by
`batch_metrics_perfile_rate.py`. HH/LL mirror H/L because the metrics script
only looks at H and L; the four-column layout matches the global limits_zjsh
schema purely for compatibility / readability.

This script does not modify any existing CSV, code, or model output — it only
writes a NEW limits CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='batch_metrics yaml (used only for params.target, params.data_root, test list)')
    p.add_argument('--quantile', type=float, default=0.95,
                   help='quantile q for τ_file = quantile(|Δx|, q). Default 0.95.')
    p.add_argument('--output', required=True,
                   help='output CSV path for per-file limits')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    params = cfg.get('params', {})
    target = params['target']
    data_root = Path(params['data_root'])
    test_files = cfg['test']

    rows = []
    print(f'Building per-file Δ-space limits at quantile={args.quantile} for target={target}')
    for name in test_files:
        df = pd.read_csv(data_root / name)
        arr = df[target].to_numpy(copy=True).astype(np.float64)
        diff = np.diff(arr)
        abs_diff = np.abs(diff)
        tau = float(np.quantile(abs_diff, args.quantile))
        rows.append({
            'filename': name,
            'HH': tau,
            'H': tau,
            'L': -tau,
            'LL': -tau,
            'n_steps': int(len(arr)),
            'mean_abs_diff': float(abs_diff.mean()),
            'std_diff': float(diff.std(ddof=0)),
        })
        print(f'  {name}: len={len(arr):5d}  '
              f'mean|Δ|={abs_diff.mean():.4f}  std(Δ)={diff.std():.4f}  '
              f'q{args.quantile:.2f}|Δ|=τ={tau:.4f}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows,
        columns=['filename', 'HH', 'H', 'L', 'LL', 'n_steps', 'mean_abs_diff', 'std_diff'])
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
