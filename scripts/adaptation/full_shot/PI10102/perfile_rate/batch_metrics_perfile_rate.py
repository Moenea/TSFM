#!/usr/bin/env python
"""
Per-file Δ-space alarm-aware batch metrics for PI10102 transition models.

A FULLY-INDEPENDENT parallel evaluator. It does NOT modify or import
utils/batch_metrics.py. The metric algebra mirrors that file, with two
substantive changes:

  1. Predictions/targets are converted to Δ space before alarm logic:
       Δpred[w, 0]   = pred[w, 0] - x_last_input[w]
       Δpred[w, t]   = pred[w, t] - pred[w, t-1]       (t ≥ 1)
       (same for true)
     For DIFF models this is equivalent (up to integration round-trip) to the
     saved pred_diff.npy; for raw models this is a fresh conversion.

  2. Alarm limits are *per-file*: each test transition gets its own (HH/H/L/LL)
     loaded from a CSV with columns [filename, HH, H, L, LL]. Limits are
     applied two-sided (|Δx| > H or |Δx| < L). Quality-gate band uses
     alarm_band_file = H_file - L_file.

Outputs go to a configurable directory (default figures/PI10102/diff/):
  - <out_dir>/metrics_C/<safe_model_name>.json   per-model metrics
  - <out_dir>/summary.csv  /  summary.md         consolidated table
  - <out_dir>/<key>.png                          per-metric bar charts
  - <out_dir>/summary_radar_C.png                radar
  - <out_dir>/limits_perfile.csv                 copy of input limits for traceability

All windowing (build_window_starts, align_window_mask), eval_steps truncation
and align_eval_to logic are identical to utils/batch_metrics.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass
plt.rcParams['figure.max_open_warning'] = 0


# ---------------------------------------------------------------------------
# Window/alignment helpers (mirrored from utils/batch_metrics.py)
# ---------------------------------------------------------------------------

def contiguous_events(mask):
    events = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(mask) - 1))
    return events


def build_window_starts(seq_len, pred_len, file_lengths):
    usable = [max(0, L - seq_len - pred_len + 1) for L in file_lengths]
    cum = np.cumsum([0] + usable)
    offsets = np.cumsum([0] + file_lengths)
    total = cum[-1]
    starts = np.zeros(total, dtype=np.int64)
    file_idx_arr = np.zeros(total, dtype=np.int64)
    for gi in range(total):
        f = int(np.searchsorted(cum[1:], gi, side='right'))
        starts[gi] = offsets[f] + (gi - cum[f]) + seq_len
        file_idx_arr[gi] = f
    return starts, file_idx_arr


def align_window_mask(window_starts, file_idx_arr, file_lengths,
                      align_seq_len, align_pred_len):
    offsets = np.cumsum([0] + list(file_lengths))
    file_off = offsets[file_idx_arr]
    file_len = np.array(file_lengths, dtype=np.int64)[file_idx_arr]
    lo = file_off + int(align_seq_len)
    hi = file_off + file_len - int(align_pred_len)
    return (window_starts >= lo) & (window_starts <= hi)


# ---------------------------------------------------------------------------
# Plotting (lifted from utils/batch_metrics.py — same look & feel)
# ---------------------------------------------------------------------------

def plot_metrics(metrics_map, tag, keys, fig_dir):
    if not metrics_map:
        return
    models_order = list(metrics_map.keys())
    for key in keys:
        vals = []
        for m in models_order:
            v = metrics_map[m].get(key, None)
            vals.append(0.0 if v is None else float(v))
        plt.figure()
        bars = plt.bar(models_order, vals, color='#293890')
        for bar, v in zip(bars, vals):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{v:.4g}', ha='center', va='bottom', fontsize=8)
        plt.title(f'{tag} (Δ-space, per-file alarm) - {key}')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(fig_dir / f'{key}.png', dpi=200)
        plt.close()


def plot_radar(metrics_map, labels, tag, fig_dir):
    if not metrics_map:
        return
    models_order = list(metrics_map.keys())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    mse_keys = {k for k in labels if k.startswith('mse_')}

    series = []
    for m in models_order:
        vals = []
        for key in labels:
            v = metrics_map.get(m, {}).get(key, None)
            vals.append(0.0 if v is None else float(v))
        vals += vals[:1]
        series.append(vals)

    if mse_keys:
        key_to_index = {k: i for i, k in enumerate(labels)}
        eps = 1e-12
        for key in mse_keys:
            idx = key_to_index[key]
            col = np.array([s[idx] for s in series], dtype=np.float64)
            inv = 1.0 / np.maximum(col, eps)
            min_v, max_v = float(np.min(inv)), float(np.max(inv))
            scores = (inv - min_v) / (max_v - min_v) if max_v > min_v else np.ones_like(inv)
            for i in range(len(series)):
                series[i][idx] = float(scores[i])

    max_per_metric = []
    for i in range(len(labels)):
        vals = [s[i] for s in series]
        max_val = max(vals) if vals else 1.0
        max_per_metric.append(max_val if max_val != 0 else 1.0)

    norm_series = []
    for i in range(len(series)):
        v = [series[i][j] / max_per_metric[j] for j in range(len(labels))]
        v += v[:1]
        norm_series.append(v)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    radar_labels = [(f'1/{k}' if k in mse_keys else k) for k in labels]
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    for i in range(1, 6):
        r = 0.98 * i / 5
        ax.plot(angles, [r] * (num_vars + 1), linewidth=0.8,
                color='gray', alpha=0.3, linestyle='--')

    colors = plt.cm.tab10.colors
    highlight_start = max(0, len(models_order) - 2)
    for i, m in enumerate(models_order):
        color = colors[i % len(colors)]
        is_hl = i >= highlight_start
        lw = 2.6 if is_hl else 1.2
        la = 1.0 if is_hl else 0.35
        fa = 0.18 if is_hl else 0.05
        ax.plot(angles, norm_series[i], color=color, linewidth=lw,
                alpha=la, marker='o',
                markersize=3.5 if is_hl else 2.5, label=m)
        ax.fill(angles, norm_series[i], color=color, alpha=fa)
        ax.scatter(angles, norm_series[i], color=color,
                   s=8 if is_hl else 4, alpha=la)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    ax.spines['polar'].set_alpha(0.3)
    ax.set_yticklabels([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=2)
    ax.set_title(f'Δ-space per-file alarm — Metrics Summary (C{tag})', pad=14, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / f'summary_radar_{tag}.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Δ-space conversion helpers
# ---------------------------------------------------------------------------

def raw_to_diff(pred_t, true_t, window_starts, true_series):
    """Convert (N, H) raw forecasts to (N, H) Δ-space forecasts.

    Δ[:, 0]   = X[:, 0] - x_last_input            (per-window scalar)
    Δ[:, t]   = X[:, t] - X[:, t-1]               for t >= 1
    where x_last_input = true_series[window_starts[w] - 1].
    Same formula applied to true_t.
    """
    last_input = true_series[window_starts - 1].astype(np.float64)
    dpred = np.empty_like(pred_t, dtype=np.float64)
    dtrue = np.empty_like(true_t, dtype=np.float64)
    dpred[:, 0] = pred_t[:, 0] - last_input
    dpred[:, 1:] = np.diff(pred_t, axis=1)
    dtrue[:, 0] = true_t[:, 0] - last_input
    dtrue[:, 1:] = np.diff(true_t, axis=1)
    return dpred, dtrue


def build_diff_alarm_series(true_series, file_lengths, file_taus):
    """Boolean array len(true_series): |Δtrue| > τ_file_for_that_step.
    Δtrue at the very first step of each file is zero (no predecessor).
    """
    alarm = np.zeros(len(true_series), dtype=bool)
    offsets = np.cumsum([0] + list(file_lengths))
    for fi, L in enumerate(file_lengths):
        a, b = offsets[fi], offsets[fi] + L
        seg = true_series[a:b]
        d = np.empty_like(seg, dtype=np.float64)
        d[0] = 0.0
        d[1:] = np.diff(seg)
        alarm[a:b] = np.abs(d) > file_taus[fi]
    return alarm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='batch_metrics yaml (used for params, model_dirs, test files)')
    p.add_argument('--limits_csv', required=True,
                   help='per-file limits CSV produced by build_limits.py '
                        '(columns: filename, HH, H, L, LL[, ...])')
    p.add_argument('--out_dir', default='./figures/PI10102/diff',
                   help='output directory for plots, summary, per-model metrics')
    return p.parse_args()


def safe_mean(arr, mask):
    return float(np.mean(arr[mask])) if np.any(mask) else None


def main():
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    params = cfg.get('params', {}) or {}
    model_dirs = cfg.get('model_dirs', [])
    if not model_dirs:
        raise SystemExit('model_dirs must be set in config')

    target = params.get('target')
    default_seq_len = params.get('seq_len')
    default_pred_len = params.get('pred_len')
    alarm_quality_rmse_factor = params.get('alarm_quality_rmse_factor', None)
    results_root = Path(params.get('results_root', './results'))
    eval_steps = params.get('eval_steps', None)
    input_clean_steps = params.get('input_clean_steps', default_seq_len)
    align_cfg = params.get('align_eval_to', None) or {}
    align_seq_len = align_cfg.get('seq_len') if align_cfg else None
    align_pred_len = align_cfg.get('pred_len') if align_cfg else None

    # --- per-file limits ---
    limits_df = pd.read_csv(args.limits_csv).set_index('filename')
    test_files = cfg.get('test', []) or []
    if not test_files:
        raise SystemExit('test files not set in config')

    # Reorder limits to match test_files order; require all to be present.
    missing = [f for f in test_files if f not in limits_df.index]
    if missing:
        raise SystemExit(f'limits CSV missing rows for: {missing}')
    limits_per_file = limits_df.loc[test_files][['HH', 'H', 'L', 'LL']].to_numpy(dtype=np.float64)
    H_per_file = limits_per_file[:, 1]
    L_per_file = limits_per_file[:, 2]
    band_per_file = H_per_file - L_per_file
    tau_per_file = np.maximum(H_per_file, -L_per_file)  # |τ| for the symmetric case

    # --- load ground-truth time series ---
    data_root = Path(params.get('data_root', '/home/aicode/sherwin/dataset/ZJSH'))
    true_series_parts = []
    file_lengths = []
    for name in test_files:
        df = pd.read_csv(data_root / name)
        if target not in df.columns:
            raise SystemExit(f'target not in test file: {name}')
        arr = df[target].to_numpy(copy=True).astype(np.float64)
        true_series_parts.append(arr)
        file_lengths.append(len(arr))
    true_series = np.concatenate(true_series_parts, axis=0)

    # --- Δ-space true alarm bool over the concatenated series (per-file τ) ---
    diff_true_alarm_series = build_diff_alarm_series(true_series, file_lengths, tau_per_file)

    print(f'target={target}  test files={len(test_files)}  '
          f'align_eval_to=({align_seq_len},{align_pred_len})  eval_steps={eval_steps}')
    print('per-file limits:')
    for i, name in enumerate(test_files):
        print(f'  [{i}] {name}: H=+{H_per_file[i]:.4f}  L={L_per_file[i]:.4f}  '
              f'band={band_per_file[i]:.4f}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics_C').mkdir(parents=True, exist_ok=True)
    # Copy the limits CSV for traceability
    try:
        limits_df.to_csv(out_dir / 'limits_perfile.csv')
    except Exception:
        pass

    metrics_all_C = {}

    for entry in model_dirs:
        model_name = entry['name']
        result_dir = results_root / entry['result_dir']
        pred_path = result_dir / 'pred.npy'
        true_path = result_dir / 'true.npy'
        if not pred_path.exists() or not true_path.exists():
            print(f'missing pred/true for {model_name}: {result_dir}')
            continue

        seq_len = entry.get('seq_len', default_seq_len)
        pred_len = entry.get('pred_len', default_pred_len)
        if seq_len is None or pred_len is None:
            raise SystemExit(
                f'seq_len/pred_len missing for {model_name} (set in params or entry)')

        pred = np.load(pred_path)
        true = np.load(true_path)
        if pred.ndim == 3:
            pred_t = pred[:, :, -1].astype(np.float64)
            true_t = true[:, :, -1].astype(np.float64)
        else:
            pred_t = pred.astype(np.float64)
            true_t = true.astype(np.float64)

        window_starts_full, file_idx_full = build_window_starts(
            seq_len, pred_len, file_lengths)

        if align_seq_len is not None and align_pred_len is not None:
            keep = align_window_mask(
                window_starts_full, file_idx_full, file_lengths,
                align_seq_len, align_pred_len)
            if pred_t.shape[0] != window_starts_full.shape[0]:
                raise SystemExit(
                    f'window count mismatch for {model_name}: '
                    f'pred has {pred_t.shape[0]} rows but expected '
                    f'{window_starts_full.shape[0]} from seq_len={seq_len}, '
                    f'pred_len={pred_len}')
            pred_t = pred_t[keep]
            true_t = true_t[keep]
            window_starts = window_starts_full[keep]
            file_idx_arr = file_idx_full[keep]
        else:
            window_starts = window_starts_full
            file_idx_arr = file_idx_full

        if eval_steps is not None and eval_steps < pred_t.shape[1]:
            pred_t = pred_t[:, :eval_steps]
            true_t = true_t[:, :eval_steps]

        pred_len_eff = int(pred_t.shape[1])
        _eff_steps = eval_steps if eval_steps is not None else pred_len_eff
        half_start = _eff_steps // 2

        # --- Δ-space conversion ---
        dpred, dtrue = raw_to_diff(pred_t, true_t, window_starts, true_series)

        # --- per-window per-step alarm in Δ space, two-sided ---
        H_w = H_per_file[file_idx_arr][:, None]  # (N, 1)
        L_w = L_per_file[file_idx_arr][:, None]  # (N, 1)
        band_w = (H_per_file - L_per_file)[file_idx_arr]  # (N,)

        pred_patch_alarm = (dpred > H_w) | (dpred < L_w)
        true_patch_alarm = (dtrue > H_w) | (dtrue < L_w)
        pred_alarm_patch = np.any(pred_patch_alarm, axis=1)
        true_alarm_patch = np.any(true_patch_alarm, axis=1)
        true_alarm_last5 = np.any(true_patch_alarm[:, half_start:], axis=1)
        pred_alarm_last5 = np.any(pred_patch_alarm[:, half_start:], axis=1)

        patch_se = (dpred - dtrue) ** 2
        patch_ae = np.abs(dpred - dtrue)
        patch_mse = np.mean(patch_se, axis=1)
        patch_rmse = np.sqrt(patch_mse)
        patch_mae = np.mean(patch_ae, axis=1)
        denom = np.maximum(np.abs(dtrue), 1e-12)
        patch_mape = np.mean(patch_ae / denom, axis=1)

        # --- Quality gate: per-window band ---
        if alarm_quality_rmse_factor is not None:
            rmse_threshold = band_w * float(alarm_quality_rmse_factor)
            pred_quality_ok = patch_rmse <= rmse_threshold
        else:
            pred_quality_ok = np.ones(len(pred_t), dtype=bool)
        pred_alarm_patch_qf = pred_alarm_patch & pred_quality_ok
        pred_alarm_last5_qf = pred_alarm_last5 & pred_quality_ok

        # --- Conditional MSE/RMSE/MAE/MAPE ---
        mse_true_alarm = safe_mean(patch_mse, true_alarm_last5)
        mse_no_true_alarm = safe_mean(patch_mse, ~true_alarm_patch)
        mse_pred_alarm = safe_mean(patch_mse, pred_alarm_patch)
        mse_no_pred_alarm = safe_mean(patch_mse, ~pred_alarm_patch)

        rmse_true_alarm = safe_mean(patch_rmse, true_alarm_last5)
        rmse_no_true_alarm = safe_mean(patch_rmse, ~true_alarm_patch)
        rmse_pred_alarm = safe_mean(patch_rmse, pred_alarm_patch)
        rmse_no_pred_alarm = safe_mean(patch_rmse, ~pred_alarm_patch)

        mae_true_alarm = safe_mean(patch_mae, true_alarm_last5)
        mae_no_true_alarm = safe_mean(patch_mae, ~true_alarm_patch)
        mae_pred_alarm = safe_mean(patch_mae, pred_alarm_patch)
        mae_no_pred_alarm = safe_mean(patch_mae, ~pred_alarm_patch)

        mape_true_alarm = safe_mean(patch_mape, true_alarm_last5)
        mape_no_true_alarm = safe_mean(patch_mape, ~true_alarm_patch)
        mape_pred_alarm = safe_mean(patch_mape, pred_alarm_patch)
        mape_no_pred_alarm = safe_mean(patch_mape, ~pred_alarm_patch)

        mse_all = safe_mean(patch_mse, np.ones(len(patch_mse), dtype=bool))
        rmse_all = safe_mean(patch_rmse, np.ones(len(patch_rmse), dtype=bool))
        mae_all = safe_mean(patch_mae, np.ones(len(patch_mae), dtype=bool))
        mape_all = safe_mean(patch_mape, np.ones(len(patch_mape), dtype=bool))

        y_true = dtrue.reshape(-1)
        y_pred = dpred.reshape(-1)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2_all = None if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

        ratio_pred_in_true = safe_mean(pred_alarm_last5.astype(float), true_alarm_last5)
        ratio_pred_in_no_true = safe_mean(pred_alarm_last5.astype(float), ~true_alarm_patch)

        # --- Event-level lead time on Δ-space true alarm series (per-file τ) ---
        true_events = contiguous_events(diff_true_alarm_series)

        def _lead_times(pred_mask):
            res = []
            for (s, _) in true_events:
                cand = np.where(
                    pred_mask
                    & (window_starts <= s)
                    & (s <= (window_starts + pred_len_eff - 1))
                )[0]
                if cand.size == 0:
                    res.append(0)
                    continue
                earliest = int(np.min(window_starts[cand]))
                res.append(int(max(0, min(s - earliest, pred_len_eff))))
            return res

        def _prognosis_err(mask):
            errs = []
            for i in range(len(pred_t)):
                if not mask[i]:
                    continue
                ts = np.where(true_patch_alarm[i])[0]
                ps = np.where(pred_patch_alarm[i])[0]
                if ts.size > 0 and ps.size > 0:
                    errs.append(abs(int(ps[0]) - int(ts[0])))
            return errs

        lead_times = _lead_times(pred_alarm_patch)
        mean_lead_time_patch = float(np.mean(lead_times)) if lead_times else None
        prognosis_errors = _prognosis_err(true_alarm_patch)
        mean_prognosis_error = float(np.mean(prognosis_errors)) if prognosis_errors else None

        # --- Clean-input variants (input window has no Δ-space alarm) ---
        input_clean = np.array([
            not np.any(diff_true_alarm_series[max(0, ws - input_clean_steps):ws])
            for ws in window_starts
        ], dtype=bool)

        true_alarm_last5_clean = true_alarm_last5 & input_clean
        true_alarm_patch_clean = true_alarm_patch & input_clean

        ratio_pred_in_true_clean = safe_mean(
            pred_alarm_last5.astype(float), true_alarm_last5_clean)
        ratio_pred_in_no_true_clean = safe_mean(
            pred_alarm_last5.astype(float), ~true_alarm_patch_clean & input_clean)
        lead_times_clean = _lead_times(pred_alarm_patch & input_clean)
        mean_lead_time_patch_clean = float(np.mean(lead_times_clean)) if lead_times_clean else None
        prognosis_errors_clean = _prognosis_err(true_alarm_patch & input_clean)
        mean_prognosis_error_clean = float(np.mean(prognosis_errors_clean)) if prognosis_errors_clean else None

        # --- Quality-filtered variants ---
        ratio_pred_in_true_qf = safe_mean(pred_alarm_last5_qf.astype(float), true_alarm_last5)
        ratio_pred_in_no_true_qf = safe_mean(pred_alarm_last5_qf.astype(float), ~true_alarm_patch)
        lead_times_qf = _lead_times(pred_alarm_patch_qf)
        mean_lead_time_patch_qf = float(np.mean(lead_times_qf)) if lead_times_qf else None
        prognosis_errors_qf = _prognosis_err(true_alarm_patch & pred_quality_ok)
        mean_prognosis_error_qf = float(np.mean(prognosis_errors_qf)) if prognosis_errors_qf else None

        # --- Clean + quality-filtered ---
        pred_alarm_patch_clean_qf = pred_alarm_patch & input_clean & pred_quality_ok
        pred_alarm_last5_clean_qf = pred_alarm_last5 & input_clean & pred_quality_ok
        ratio_pred_in_true_clean_qf = safe_mean(
            pred_alarm_last5_clean_qf.astype(float), true_alarm_last5_clean)
        ratio_pred_in_no_true_clean_qf = safe_mean(
            pred_alarm_last5_clean_qf.astype(float),
            ~true_alarm_patch_clean & input_clean)
        lead_times_clean_qf = _lead_times(pred_alarm_patch_clean_qf)
        mean_lead_time_patch_clean_qf = float(np.mean(lead_times_clean_qf)) if lead_times_clean_qf else None
        prognosis_errors_clean_qf = _prognosis_err(true_alarm_patch & input_clean & pred_quality_ok)
        mean_prognosis_error_clean_qf = float(np.mean(prognosis_errors_clean_qf)) if prognosis_errors_clean_qf else None

        metrics_C = {
            'mse_true_alarm_patch': mse_true_alarm,
            'mse_no_true_alarm_patch': mse_no_true_alarm,
            'mse_pred_alarm_patch': mse_pred_alarm,
            'mse_no_pred_alarm_patch': mse_no_pred_alarm,
            'rmse_true_alarm_patch': rmse_true_alarm,
            'rmse_no_true_alarm_patch': rmse_no_true_alarm,
            'rmse_pred_alarm_patch': rmse_pred_alarm,
            'rmse_no_pred_alarm_patch': rmse_no_pred_alarm,
            'mae_true_alarm_patch': mae_true_alarm,
            'mae_no_true_alarm_patch': mae_no_true_alarm,
            'mae_pred_alarm_patch': mae_pred_alarm,
            'mae_no_pred_alarm_patch': mae_no_pred_alarm,
            'mape_true_alarm_patch': mape_true_alarm,
            'mape_no_true_alarm_patch': mape_no_true_alarm,
            'mape_pred_alarm_patch': mape_pred_alarm,
            'mape_no_pred_alarm_patch': mape_no_pred_alarm,
            'ratio_pred_in_true_alarm_patches': ratio_pred_in_true,
            'ratio_pred_in_no_true_alarm_patches': ratio_pred_in_no_true,
            'mean_lead_time_patch': mean_lead_time_patch,
            'mean_prognosis_error': mean_prognosis_error,
            'ratio_pred_in_true_alarm_patches_clean': ratio_pred_in_true_clean,
            'ratio_pred_in_no_true_alarm_patches_clean': ratio_pred_in_no_true_clean,
            'mean_lead_time_patch_clean': mean_lead_time_patch_clean,
            'mean_prognosis_error_clean': mean_prognosis_error_clean,
            'ratio_pred_in_true_alarm_patches_qf': ratio_pred_in_true_qf,
            'ratio_pred_in_no_true_alarm_patches_qf': ratio_pred_in_no_true_qf,
            'mean_lead_time_patch_qf': mean_lead_time_patch_qf,
            'mean_prognosis_error_qf': mean_prognosis_error_qf,
            'ratio_pred_in_true_alarm_patches_clean_qf': ratio_pred_in_true_clean_qf,
            'ratio_pred_in_no_true_alarm_patches_clean_qf': ratio_pred_in_no_true_clean_qf,
            'mean_lead_time_patch_clean_qf': mean_lead_time_patch_clean_qf,
            'mean_prognosis_error_clean_qf': mean_prognosis_error_clean_qf,
            'n_pred_alarm_patches': int(np.sum(pred_alarm_patch)),
            'n_pred_alarm_patches_qf': int(np.sum(pred_alarm_patch_qf)),
            'n_quality_rejected': int(np.sum(pred_alarm_patch & ~pred_quality_ok)),
            'n_true_alarm_patches': int(np.sum(true_alarm_patch)),
            'tp_window': int(np.sum(pred_alarm_last5 & true_alarm_last5)),
            'fp_window': int(np.sum(pred_alarm_last5 & ~true_alarm_last5)),
            'fn_window': int(np.sum(~pred_alarm_last5 & true_alarm_last5)),
            'tn_window': int(np.sum(~pred_alarm_last5 & ~true_alarm_last5)),
            'tp_window_clean': int(np.sum(pred_alarm_last5 & true_alarm_last5_clean & input_clean)),
            'fp_window_clean': int(np.sum(pred_alarm_last5 & ~true_alarm_last5_clean & input_clean)),
            'fn_window_clean': int(np.sum(~pred_alarm_last5 & true_alarm_last5_clean & input_clean)),
            'tn_window_clean': int(np.sum(~pred_alarm_last5 & ~true_alarm_last5_clean & input_clean)),
            'tp_window_qf': int(np.sum(pred_alarm_last5_qf & true_alarm_last5)),
            'fp_window_qf': int(np.sum(pred_alarm_last5_qf & ~true_alarm_last5)),
            'fn_window_qf': int(np.sum(~pred_alarm_last5_qf & true_alarm_last5)),
            'tn_window_qf': int(np.sum(~pred_alarm_last5_qf & ~true_alarm_last5)),
            'tp_window_clean_qf': int(np.sum(pred_alarm_last5_clean_qf & true_alarm_last5_clean & input_clean)),
            'fp_window_clean_qf': int(np.sum(pred_alarm_last5_clean_qf & ~true_alarm_last5_clean & input_clean)),
            'fn_window_clean_qf': int(np.sum(~pred_alarm_last5_clean_qf & true_alarm_last5_clean & input_clean)),
            'tn_window_clean_qf': int(np.sum(~pred_alarm_last5_clean_qf & ~true_alarm_last5_clean & input_clean)),
            'mse_all_patches': mse_all,
            'rmse_all_patches': rmse_all,
            'mae_all_patches': mae_all,
            'mape_all_patches': mape_all,
            'r2_all_points': r2_all,
        }

        safe_name = model_name.replace('/', '_')
        (out_dir / 'metrics_C' / f'{safe_name}.json').write_text(
            json.dumps(metrics_C, indent=2))
        metrics_all_C[model_name] = metrics_C
        print(f'[{model_name}] N={len(pred_t):5d}  mse_all={mse_all:.4g}  '
              f'n_true_alarm={int(np.sum(true_alarm_patch))}  '
              f'n_pred_alarm={int(np.sum(pred_alarm_patch))}')

    # --- Plots ---
    plot_keys_c = [
        'mse_all_patches',
        'mse_true_alarm_patch', 'mse_no_true_alarm_patch',
        'mse_pred_alarm_patch', 'mse_no_pred_alarm_patch',
        'ratio_pred_in_true_alarm_patches', 'ratio_pred_in_no_true_alarm_patches',
        'mean_lead_time_patch', 'mean_prognosis_error',
        'ratio_pred_in_true_alarm_patches_clean', 'ratio_pred_in_no_true_alarm_patches_clean',
        'mean_lead_time_patch_clean', 'mean_prognosis_error_clean',
        'ratio_pred_in_true_alarm_patches_qf', 'ratio_pred_in_no_true_alarm_patches_qf',
        'mean_lead_time_patch_qf', 'mean_prognosis_error_qf',
        'ratio_pred_in_true_alarm_patches_clean_qf', 'ratio_pred_in_no_true_alarm_patches_clean_qf',
        'mean_lead_time_patch_clean_qf', 'mean_prognosis_error_clean_qf',
    ]
    plot_keys_radar = [
        'mse_true_alarm_patch', 'mse_no_true_alarm_patch',
        'mse_pred_alarm_patch', 'mse_no_pred_alarm_patch',
        'ratio_pred_in_true_alarm_patches_clean_qf',
        'ratio_pred_in_no_true_alarm_patches_clean_qf',
        'mean_lead_time_patch_clean_qf', 'mean_prognosis_error_clean_qf',
    ]

    plot_metrics(metrics_all_C, 'C', plot_keys_c, out_dir)
    plot_radar(metrics_all_C, plot_keys_radar, 'C', out_dir)

    # --- Summary CSV ---
    summary_cols = [
        'ratio_pred_in_true_alarm_patches', 'ratio_pred_in_no_true_alarm_patches',
        'mean_lead_time_patch', 'mean_prognosis_error',
        'ratio_pred_in_true_alarm_patches_clean', 'ratio_pred_in_no_true_alarm_patches_clean',
        'mean_lead_time_patch_clean', 'mean_prognosis_error_clean',
        'ratio_pred_in_true_alarm_patches_qf', 'ratio_pred_in_no_true_alarm_patches_qf',
        'mean_lead_time_patch_qf', 'mean_prognosis_error_qf',
        'ratio_pred_in_true_alarm_patches_clean_qf', 'ratio_pred_in_no_true_alarm_patches_clean_qf',
        'mean_lead_time_patch_clean_qf', 'mean_prognosis_error_clean_qf',
        'n_pred_alarm_patches', 'n_pred_alarm_patches_qf',
        'n_true_alarm_patches', 'n_quality_rejected',
        'mse_all_patches',
        'rmse_true_alarm_patch', 'rmse_no_true_alarm_patch',
        'rmse_pred_alarm_patch', 'rmse_no_pred_alarm_patch', 'rmse_all_patches',
        'mae_true_alarm_patch', 'mae_no_true_alarm_patch',
        'mae_pred_alarm_patch', 'mae_no_pred_alarm_patch', 'mae_all_patches',
        'tp_window', 'fp_window', 'tn_window', 'fn_window',
        'tp_window_clean', 'fp_window_clean', 'tn_window_clean', 'fn_window_clean',
        'tp_window_qf', 'fp_window_qf', 'tn_window_qf', 'fn_window_qf',
        'tp_window_clean_qf', 'fp_window_clean_qf', 'tn_window_clean_qf', 'fn_window_clean_qf',
    ]
    rows = []
    for entry in model_dirs:
        mc = metrics_all_C.get(entry['name'])
        if not mc:
            continue
        row = {'model': entry['name']}
        for k in summary_cols:
            row[k] = mc.get(k, None)
        rows.append(row)
    summary_df = pd.DataFrame(rows, columns=['model'] + summary_cols)
    summary_df.to_csv(out_dir / 'summary.csv', index=False)
    try:
        (out_dir / 'summary.md').write_text(summary_df.to_markdown(index=False), encoding='utf-8')
    except Exception:
        pass
    print(f'wrote {out_dir / "summary.csv"}')
    print(f'wrote per-model metrics under {out_dir / "metrics_C"}')


if __name__ == '__main__':
    main()
