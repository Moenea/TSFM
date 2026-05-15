#!/usr/bin/env python
"""
Gate-ensemble fusion of two PI10102 prediction sources.

Combines source A (e.g., DIFF-to-ABS Timer-XL) and source B (e.g., raw Timer-XL)
into a single fused prediction stream evaluated in the same raw PI10102 space.

Three methods:
  tier1  per-step static convex weight alpha(t) in [0,1]^eval_steps,
         closed-form least squares.
  tier2  context-only gate: small MLP takes input-window features ->
         sigma(t) in [0,1]^eval_steps; fused = sigma * A + (1 - sigma) * B.
  tier3  stacking MLP: takes [features, A(1..15), B(1..15)] -> outputs 15-step
         forecast directly (no convex constraint).

Train/eval split:
  The existing pipeline only saves base-model predictions on the test split
  (4 transitions). To avoid leakage we use leave-one-transition-out CV: fit
  the gate on 3 transitions, predict on the 4th, repeat 4 times. The final
  pred.npy is the concatenated out-of-fold predictions for all aligned
  windows.

Output:
  results/ensemble_<output_name>_test_0/
      pred.npy   shape (N_aligned, 96)  -- first eval_steps cols = fused output,
                                          remaining cols = source A passthrough
      true.npy   shape (N_aligned, 96)
      fit_log.json   per-fold metrics, alphas, hyperparameters

Both yaml files (raw + diff) are searched for the source entries, so any
combination of raw + DIFF-to-ABS pairs is supported.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


TSFM_ROOT = Path(__file__).resolve().parents[5]


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def find_entry(name, cfgs):
    """Search a list of yaml configs for the first entry matching name."""
    for cfg in cfgs:
        for entry in cfg.get('model_dirs', []):
            if entry.get('name') == name:
                return entry
    raise KeyError(f'model entry {name!r} not found in any of {len(cfgs)} configs')


def build_window_starts(seq_len, pred_len, file_lengths):
    """Same logic as in batch_metrics.py / notebook."""
    usable = [max(0, L - seq_len - pred_len + 1) for L in file_lengths]
    cum = np.cumsum([0] + usable)
    offsets = np.cumsum([0] + file_lengths)
    starts = np.zeros(cum[-1], dtype=np.int64)
    file_idx = np.zeros(cum[-1], dtype=np.int64)
    for gi in range(cum[-1]):
        f = int(np.searchsorted(cum[1:], gi, side='right'))
        starts[gi] = offsets[f] + (gi - cum[f]) + seq_len
        file_idx[gi] = f
    return starts, file_idx


def align_mask(window_starts, file_idx_arr, file_lengths, asl, apl):
    offsets = np.cumsum([0] + list(file_lengths))
    file_off = offsets[file_idx_arr]
    file_len = np.array(file_lengths, dtype=np.int64)[file_idx_arr]
    return (window_starts >= file_off + asl) & (window_starts <= file_off + file_len - apl)


def load_aligned(entry, results_root, file_lengths, eval_steps, asl, apl):
    """Load pred/true.npy and return aligned arrays sliced to first eval_steps."""
    p_dir = Path(results_root) / entry['result_dir']
    pred = np.load(p_dir / 'pred.npy')
    true = np.load(p_dir / 'true.npy')
    if pred.ndim == 3:
        pred = pred[:, :, -1]
    if true.ndim == 3:
        true = true[:, :, -1]
    sl = int(entry['seq_len']); pl = int(entry['pred_len'])
    ws, fi = build_window_starts(sl, pl, file_lengths)
    if pred.shape[0] != ws.shape[0]:
        raise ValueError(
            f'{entry["name"]}: pred has {pred.shape[0]} rows but expected {ws.shape[0]} '
            f'(sl={sl}, pl={pl}, file_lengths={file_lengths})'
        )
    keep = align_mask(ws, fi, file_lengths, asl, apl)
    return (
        pred[keep][:, :eval_steps].astype(np.float64),
        true[keep][:, :eval_steps].astype(np.float64),
        ws[keep],
        fi[keep],
    )


def load_true_series(test_files, data_root, target):
    series = []
    lens = []
    for name in test_files:
        df = pd.read_csv(Path(data_root) / name)
        arr = df[target].to_numpy(copy=True).astype(np.float64)
        series.append(arr); lens.append(len(arr))
    return np.concatenate(series, axis=0), lens


def build_context_features(true_series, window_starts, asl):
    """Per-window features computed from the input window (last asl steps).

    8 features:
      - last_val
      - one-step diff (last - prev)
      - mean over last 96
      - std over last 96
      - slope of linear fit over last 96
      - range over last asl (max - min)
      - distance above L limit (last_val - 160)
      - distance below H limit (260 - last_val)
    """
    H, L = 260.0, 160.0
    short = 96
    n = window_starts.shape[0]
    feats = np.zeros((n, 8), dtype=np.float64)
    x96 = np.arange(short, dtype=np.float64)
    x96_mean = x96.mean(); x96_var = ((x96 - x96_mean) ** 2).sum()
    for i, ws in enumerate(window_starts):
        last = true_series[ws - 1]
        prev = true_series[ws - 2] if ws >= 2 else last
        win_short = true_series[ws - short:ws]
        win_long = true_series[ws - asl:ws]
        # slope: (sum((x - xbar)(y - ybar))) / (sum((x - xbar)^2))
        y_mean = win_short.mean()
        slope = float(((x96 - x96_mean) * (win_short - y_mean)).sum() / x96_var)
        feats[i] = [
            last,
            last - prev,
            win_short.mean(),
            win_short.std(),
            slope,
            float(win_long.max() - win_long.min()),
            last - L,
            H - last,
        ]
    return feats


# ----------------------- Method implementations -----------------------

def fit_predict_tier1(pa_tr, pb_tr, t_tr, pa_te, pb_te, eval_steps):
    """Per-step closed-form alpha(t) in [0,1]."""
    alphas = np.zeros(eval_steps)
    for k in range(eval_steps):
        r = pa_tr[:, k] - pb_tr[:, k]
        s = t_tr[:, k] - pb_tr[:, k]
        denom = float((r * r).sum())
        if denom < 1e-12:
            alphas[k] = 0.5
        else:
            alphas[k] = float(np.clip((r * s).sum() / denom, 0.0, 1.0))
    fused = alphas[None, :] * pa_te + (1.0 - alphas[None, :]) * pb_te
    return fused, {'alphas': alphas.tolist()}


class GateMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class StackMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(model, X, Y, *, epochs, lr, weight_decay, device, loss_fn):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    model.to(device).train()
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, Yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


def fit_predict_tier2(feats_tr, pa_tr, pb_tr, t_tr,
                      feats_te, pa_te, pb_te,
                      *, eval_steps, hidden, epochs, lr, weight_decay, seed, device):
    """Context-only gate: features -> sigma(t) -> fused = sigma*A + (1-sigma)*B."""
    torch.manual_seed(seed)
    mu = feats_tr.mean(axis=0); sigma = feats_tr.std(axis=0) + 1e-6
    f_tr = (feats_tr - mu) / sigma
    f_te = (feats_te - mu) / sigma

    model = GateMLP(in_dim=f_tr.shape[1], hidden=hidden, out_dim=eval_steps)

    pa_t = torch.tensor(pa_tr, dtype=torch.float32, device=device)
    pb_t = torch.tensor(pb_tr, dtype=torch.float32, device=device)
    yt_t = torch.tensor(t_tr, dtype=torch.float32, device=device)

    def loss_fn(sig, _):
        y_pred = sig * pa_t + (1.0 - sig) * pb_t
        return ((y_pred - yt_t) ** 2).mean()

    losses = _train_mlp(model, f_tr, t_tr,
                        epochs=epochs, lr=lr, weight_decay=weight_decay,
                        device=device, loss_fn=loss_fn)

    with torch.no_grad():
        sig_te = model(torch.tensor(f_te, dtype=torch.float32, device=device)).cpu().numpy()
    fused = sig_te * pa_te + (1.0 - sig_te) * pb_te
    return fused, {
        'final_loss': losses[-1],
        'sigma_mean_per_step': sig_te.mean(axis=0).tolist(),
    }


def fit_predict_tier3(feats_tr, pa_tr, pb_tr, t_tr,
                      feats_te, pa_te, pb_te,
                      *, eval_steps, hidden, epochs, lr, weight_decay, seed, device):
    """Stacking: MLP([features, A, B]) -> 15-step prediction."""
    torch.manual_seed(seed)
    X_tr = np.concatenate([feats_tr, pa_tr, pb_tr], axis=1)
    X_te = np.concatenate([feats_te, pa_te, pb_te], axis=1)
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0) + 1e-6
    Xn_tr = (X_tr - mu) / sigma
    Xn_te = (X_te - mu) / sigma

    y_mu = t_tr.mean(axis=0); y_sig = t_tr.std(axis=0) + 1e-6
    Y_n = (t_tr - y_mu) / y_sig

    model = StackMLP(in_dim=Xn_tr.shape[1], hidden=hidden, out_dim=eval_steps)
    losses = _train_mlp(model, Xn_tr, Y_n,
                        epochs=epochs, lr=lr, weight_decay=weight_decay,
                        device=device, loss_fn=nn.MSELoss())
    with torch.no_grad():
        y_n = model(torch.tensor(Xn_te, dtype=torch.float32, device=device)).cpu().numpy()
    fused = y_n * y_sig + y_mu
    return fused, {'final_loss': losses[-1]}


# ----------------------- Driver -----------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_a_name', required=True,
                   help='yaml entry name for source A (e.g. Timer-XL-MS-DIFF)')
    p.add_argument('--source_b_name', required=True,
                   help='yaml entry name for source B (e.g. Timer-XL-MS)')
    p.add_argument('--method', required=True, choices=['tier1', 'tier2', 'tier3'])
    p.add_argument('--output_name', required=True,
                   help='label embedded in result_dir, e.g. Gate-T1-TXL-MS')
    p.add_argument('--config_diff',
                   default=str(TSFM_ROOT / 'setting' / 'batch_metrics_zjsh_pi10102ts_diff.yaml'))
    p.add_argument('--config_raw',
                   default=str(TSFM_ROOT / 'setting' / 'batch_metrics_zjsh_pi10102ts.yaml'))
    p.add_argument('--results_root', default=str(TSFM_ROOT / 'results'))
    p.add_argument('--eval_steps', type=int, default=15)
    p.add_argument('--align_seq_len', type=int, default=768)
    p.add_argument('--align_pred_len', type=int, default=96)
    p.add_argument('--out_pred_len', type=int, default=96,
                   help='length of saved pred.npy 2nd dim (eval_steps fused, rest passthrough from A)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()

    cfg_diff = load_cfg(args.config_diff)
    cfg_raw = load_cfg(args.config_raw)
    params = cfg_diff.get('params', {})
    data_root = params['data_root']
    target = params['target']
    test_files = cfg_diff['test']

    # Search raw yaml first: it carries both raw names (e.g. Timer-XL-MS) and
    # explicitly-suffixed DIFF names (e.g. Timer-XL-MS-DIFF), so it always
    # disambiguates. The diff yaml reuses base names like Timer-XL-MS for DIFF
    # variants, which would shadow the raw entries if searched first.
    entry_a = find_entry(args.source_a_name, [cfg_raw, cfg_diff])
    entry_b = find_entry(args.source_b_name, [cfg_raw, cfg_diff])

    true_series, file_lengths = load_true_series(test_files, data_root, target)

    pa, ta, ws, fi = load_aligned(entry_a, args.results_root, file_lengths,
                                  args.eval_steps, args.align_seq_len, args.align_pred_len)
    pb, tb, ws_b, fi_b = load_aligned(entry_b, args.results_root, file_lengths,
                                      args.eval_steps, args.align_seq_len, args.align_pred_len)
    if not np.array_equal(ws, ws_b) or not np.array_equal(fi, fi_b):
        raise ValueError('source A and B have different aligned-window indexing — '
                         'cannot fuse without re-alignment')
    if not np.allclose(ta, tb):
        diff = np.abs(ta - tb).max()
        if diff > 1e-3:
            print(f'WARN: source A/B true.npy differ by max {diff:.4g}; using source A as ground truth')

    N = pa.shape[0]
    feats = build_context_features(true_series, ws, args.align_seq_len)
    print(f'aligned windows: {N}  feats: {feats.shape}  pa: {pa.shape}  pb: {pb.shape}')

    fused_oof = np.zeros_like(pa)
    fold_info = []
    n_files = len(file_lengths)
    for fold in range(n_files):
        test_mask = fi == fold
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue
        if args.method == 'tier1':
            fused_te, info = fit_predict_tier1(
                pa[train_mask], pb[train_mask], ta[train_mask],
                pa[test_mask], pb[test_mask], args.eval_steps)
        elif args.method == 'tier2':
            fused_te, info = fit_predict_tier2(
                feats[train_mask], pa[train_mask], pb[train_mask], ta[train_mask],
                feats[test_mask], pa[test_mask], pb[test_mask],
                eval_steps=args.eval_steps, hidden=args.hidden, epochs=args.epochs,
                lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
                device=args.device)
        else:
            fused_te, info = fit_predict_tier3(
                feats[train_mask], pa[train_mask], pb[train_mask], ta[train_mask],
                feats[test_mask], pa[test_mask], pb[test_mask],
                eval_steps=args.eval_steps, hidden=args.hidden, epochs=args.epochs,
                lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
                device=args.device)
        fused_oof[test_mask] = fused_te
        info.update({
            'fold': fold,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'fold_mse_vs_A': float(((pa[test_mask] - ta[test_mask]) ** 2).mean()),
            'fold_mse_vs_B': float(((pb[test_mask] - ta[test_mask]) ** 2).mean()),
            'fold_mse_fused': float(((fused_te - ta[test_mask]) ** 2).mean()),
        })
        fold_info.append(info)
        print(f'  fold {fold}: n_tr={info["n_train"]:5d}  n_te={info["n_test"]:5d}  '
              f'mse(A)={info["fold_mse_vs_A"]:.4f}  mse(B)={info["fold_mse_vs_B"]:.4f}  '
              f'mse(fused)={info["fold_mse_fused"]:.4f}')

    # Build full-length pred.npy (out_pred_len, default 96) - first eval_steps are fused,
    # remaining columns are source A passthrough for compatibility.
    out_pred_len = args.out_pred_len
    pred_full = np.zeros((N, out_pred_len), dtype=np.float64)
    pred_full[:, :args.eval_steps] = fused_oof
    if out_pred_len > args.eval_steps:
        # Re-load source A full-horizon pred (without eval_steps slicing) for passthrough.
        full_a = np.load(Path(args.results_root) / entry_a['result_dir'] / 'pred.npy')
        if full_a.ndim == 3:
            full_a = full_a[:, :, -1]
        sl = int(entry_a['seq_len']); pl = int(entry_a['pred_len'])
        ws_full, fi_full = build_window_starts(sl, pl, file_lengths)
        keep_full = align_mask(ws_full, fi_full, file_lengths, args.align_seq_len, args.align_pred_len)
        a_aligned = full_a[keep_full]
        pred_full[:, args.eval_steps:out_pred_len] = a_aligned[:, args.eval_steps:out_pred_len]

    # True.npy: pull full-horizon true from source A.
    full_at = np.load(Path(args.results_root) / entry_a['result_dir'] / 'true.npy')
    if full_at.ndim == 3:
        full_at = full_at[:, :, -1]
    sl = int(entry_a['seq_len']); pl = int(entry_a['pred_len'])
    ws_full, fi_full = build_window_starts(sl, pl, file_lengths)
    keep_full = align_mask(ws_full, fi_full, file_lengths, args.align_seq_len, args.align_pred_len)
    true_full = full_at[keep_full][:, :out_pred_len]

    out_dir = Path(args.results_root) / f'ensemble_{args.output_name}_test_0'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'pred.npy', pred_full.astype(np.float32))
    np.save(out_dir / 'true.npy', true_full.astype(np.float32))

    overall_mse_a = float(((pa - ta) ** 2).mean())
    overall_mse_b = float(((pb - ta) ** 2).mean())
    overall_mse_fused = float(((fused_oof - ta) ** 2).mean())
    log = {
        'method': args.method,
        'source_a': args.source_a_name,
        'source_b': args.source_b_name,
        'output_name': args.output_name,
        'eval_steps': args.eval_steps,
        'align_seq_len': args.align_seq_len,
        'align_pred_len': args.align_pred_len,
        'n_aligned': int(N),
        'hyperparameters': {
            'hidden': args.hidden,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'seed': args.seed,
        },
        'overall_mse_eval_steps': {
            'source_a': overall_mse_a,
            'source_b': overall_mse_b,
            'fused_oof': overall_mse_fused,
        },
        'folds': fold_info,
    }
    (out_dir / 'fit_log.json').write_text(json.dumps(log, indent=2, ensure_ascii=False))

    print('\nOverall OOF MSE (first {} steps):'.format(args.eval_steps))
    print(f'  source A ({args.source_a_name}): {overall_mse_a:.4f}')
    print(f'  source B ({args.source_b_name}): {overall_mse_b:.4f}')
    print(f'  fused ({args.method})         : {overall_mse_fused:.4f}')
    print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
