"""
batch_metrics.py — Window-level (Category C) metrics for Timer-XL results.

Adapted from MyTimeXer/utils/batch_metrics.py. Computes alarm-aware metrics
on pred.npy / true.npy saved by exp_forecast.py, including:
  - Per-window MSE/RMSE/MAE/MAPE (overall + split by alarm state)
  - Window-level TP/FP/FN/TN confusion matrix
  - Lead time & prognosis error
  - Clean-input / quality-filtered variants
  - R^2
  - Summary CSV, bar charts, and radar plot

Usage:
    python utils/batch_metrics.py --config path/to/config.yaml
"""
import json
import argparse
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
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
# Helper functions (ported from MyTimeXer)
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
    cum_windows = np.cumsum([0] + usable)
    offsets = np.cumsum([0] + file_lengths)
    total_windows = cum_windows[-1]
    starts = np.zeros(total_windows, dtype=np.int64)
    for global_i in range(total_windows):
        file_idx = np.searchsorted(cum_windows[1:], global_i, side='right')
        local_i = global_i - cum_windows[file_idx]
        base = offsets[file_idx]
        starts[global_i] = base + local_i + seq_len
    return starts


# ---------------------------------------------------------------------------
# Plotting
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
        plt.title(f'{tag} - {key}')
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

    grid_levels = 5
    grid_max = 0.98
    for i in range(1, grid_levels + 1):
        r = grid_max * i / grid_levels
        ax.plot(angles, [r] * (num_vars + 1), linewidth=0.8,
                color='gray', alpha=0.3, linestyle='--')

    colors = plt.cm.tab10.colors
    highlight_start = max(0, len(models_order) - 2)
    for i, m in enumerate(models_order):
        color = colors[i % len(colors)]
        is_highlight = i >= highlight_start
        lw = 2.6 if is_highlight else 1.2
        line_alpha = 1.0 if is_highlight else 0.35
        fill_alpha = 0.18 if is_highlight else 0.05
        ax.plot(angles, norm_series[i], color=color, linewidth=lw,
                alpha=line_alpha, marker='o',
                markersize=3.5 if is_highlight else 2.5, label=m)
        ax.fill(angles, norm_series[i], color=color, alpha=fill_alpha)
        ax.scatter(angles, norm_series[i], color=color,
                   s=8 if is_highlight else 4, alpha=line_alpha)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    ax.spines['polar'].set_alpha(0.3)
    ax.set_yticklabels([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=2)
    ax.set_title(f'Metrics Summary (C{tag})', pad=14, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / f'summary_radar_{tag}.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--summary-suffix', type=str, default='')
    parser.add_argument('--figure-suffix', type=str, default='')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    params = cfg.get('params', {}) or {}

    # model_dirs: list of {name: display_name, result_dir: dirname_under_results}
    model_dirs = cfg.get('model_dirs', [])
    if not model_dirs:
        raise SystemExit('model_dirs must be set in config')

    target = params.get('target', '0202B_PCA101A')
    seq_len = params.get('seq_len')
    pred_len = params.get('pred_len')
    alarm_quality_rmse_factor = params.get('alarm_quality_rmse_factor', None)
    limit_csv_path = params.get('limit_csv_path', '')
    results_root = Path(params.get('results_root', './results'))

    eval_steps = params.get('eval_steps', None)
    input_clean_steps = params.get('input_clean_steps', seq_len)

    if not limit_csv_path:
        raise SystemExit('limit_csv_path must be set')
    if seq_len is None or pred_len is None:
        raise SystemExit('seq_len and pred_len must be set')

    df_limit = pd.read_csv(limit_csv_path, index_col=0)
    if target not in df_limit.index:
        raise SystemExit(f'target not found in limit_csv_path: {target}')
    high = float(df_limit.loc[target].iloc[1])
    low = float(df_limit.loc[target].iloc[2])
    alarm_band = high - low

    # Load ground-truth time series from test CSV files
    test_files = cfg.get('test', []) or []
    if not test_files:
        raise SystemExit('test files not set in config')
    data_root = Path(params.get('data_root', '/home/aicode/sherwin/dataset/JJ'))
    true_series = []
    file_lengths = []
    for name in test_files:
        df = pd.read_csv(data_root / name)
        if target not in df.columns:
            raise SystemExit(f'target not in test file: {name}')
        arr = df[target].to_numpy(copy=True)
        true_series.append(arr)
        file_lengths.append(len(arr))
    true_series = np.concatenate(true_series, axis=0)

    print(f'target={target}  high={high}  low={low}  alarm_band={alarm_band}')
    print(f'test series length={len(true_series)}  seq_len={seq_len}  pred_len={pred_len}')

    metrics_all_C = {}

    for entry in model_dirs:
        model_name = entry['name']
        result_dir = results_root / entry['result_dir']
        pred_path = result_dir / 'pred.npy'
        true_path = result_dir / 'true.npy'
        if not pred_path.exists() or not true_path.exists():
            print(f'missing pred/true for {model_name}: {result_dir}')
            continue

        pred = np.load(pred_path)
        true = np.load(true_path)

        # Normalize to 2D: (N, pred_len) — target column only
        if pred.ndim == 3:
            pred_t = pred[:, :, -1]
            true_t = true[:, :, -1]
        else:
            pred_t = pred
            true_t = true

        # Truncate to first eval_steps per window if configured
        if eval_steps is not None and eval_steps < pred_t.shape[1]:
            pred_t = pred_t[:, :eval_steps]
            true_t = true_t[:, :eval_steps]

        pred_len_eff = int(pred_t.shape[1])

        # --- Window-level error metrics ---
        pred_patch_alarm = (pred_t > high) | (pred_t < low)
        true_patch_alarm = (true_t > high) | (true_t < low)
        pred_alarm_patch = np.any(pred_patch_alarm, axis=1)
        true_alarm_patch = np.any(true_patch_alarm, axis=1)
        # Use eval_steps as the effective prediction length for half_start so that
        # when pred_len=96 but only eval_steps=15 are the actual useful steps,
        # we look at the latter half of those 15 steps (not 96//2=48 which would
        # be beyond the actual predictions).
        _eff_steps = eval_steps if eval_steps is not None else pred_len_eff
        half_start = _eff_steps // 2
        true_alarm_last5 = np.any(true_patch_alarm[:, half_start:], axis=1)
        pred_alarm_last5 = np.any(pred_patch_alarm[:, half_start:], axis=1)

        patch_se = (pred_t - true_t) ** 2
        patch_ae = np.abs(pred_t - true_t)
        patch_mse = np.mean(patch_se, axis=1)
        patch_rmse = np.sqrt(patch_mse)
        patch_mae = np.mean(patch_ae, axis=1)
        denom = np.maximum(np.abs(true_t), 1e-12)
        patch_mape = np.mean(patch_ae / denom, axis=1)

        # --- Quality gate ---
        if alarm_quality_rmse_factor is not None and alarm_band > 0:
            rmse_threshold = alarm_band * float(alarm_quality_rmse_factor)
            pred_quality_ok = patch_rmse <= rmse_threshold
        else:
            pred_quality_ok = np.ones(len(pred_t), dtype=bool)
        pred_alarm_patch_qf = pred_alarm_patch & pred_quality_ok
        pred_alarm_last5_qf = pred_alarm_last5 & pred_quality_ok

        # --- Conditional MSE/RMSE/MAE/MAPE ---
        def safe_mean(arr, mask):
            return float(np.mean(arr[mask])) if np.any(mask) else None

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

        # R^2
        y_true = true_t.reshape(-1)
        y_pred = pred_t.reshape(-1)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2_all = None if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

        # --- Alarm ratios ---
        ratio_pred_in_true = safe_mean(
            pred_alarm_last5.astype(float), true_alarm_last5)
        ratio_pred_in_no_true = safe_mean(
            pred_alarm_last5.astype(float), ~true_alarm_patch)

        # --- Event-level lead time ---
        window_starts = build_window_starts(seq_len, pred_len, file_lengths)
        true_alarm_series = (true_series > high) | (true_series < low)
        true_events = contiguous_events(true_alarm_series)

        lead_times = []
        for (s, _) in true_events:
            candidates = np.where(
                pred_alarm_patch
                & (window_starts <= s)
                & (s <= (window_starts + pred_len_eff - 1))
            )[0]
            if candidates.size == 0:
                lead_times.append(0)
                continue
            earliest_start = int(np.min(window_starts[candidates]))
            lead = max(0, min(s - earliest_start, pred_len_eff))
            lead_times.append(int(lead))
        mean_lead_time_patch = float(np.mean(lead_times)) if lead_times else None

        # --- Prognosis error ---
        prognosis_errors = []
        for i in range(len(pred_t)):
            if not true_alarm_patch[i]:
                continue
            true_steps = np.where(true_patch_alarm[i])[0]
            pred_steps = np.where(pred_patch_alarm[i])[0]
            if true_steps.size > 0 and pred_steps.size > 0:
                prognosis_errors.append(abs(int(pred_steps[0]) - int(true_steps[0])))
        mean_prognosis_error = float(np.mean(prognosis_errors)) if prognosis_errors else None

        # --- Clean-input variants ---
        input_clean = np.array([
            not np.any(true_alarm_series[max(0, ws - input_clean_steps):ws])
            for ws in window_starts
        ], dtype=bool)

        true_alarm_last5_clean = true_alarm_last5 & input_clean
        true_alarm_patch_clean = true_alarm_patch & input_clean

        ratio_pred_in_true_clean = safe_mean(
            pred_alarm_last5.astype(float), true_alarm_last5_clean)
        ratio_pred_in_no_true_clean = safe_mean(
            pred_alarm_last5.astype(float), ~true_alarm_patch_clean & input_clean)

        lead_times_clean = []
        for (s, _) in true_events:
            candidates = np.where(
                pred_alarm_patch & input_clean
                & (window_starts <= s) & (s <= (window_starts + pred_len_eff - 1))
            )[0]
            if candidates.size == 0:
                lead_times_clean.append(0)
                continue
            earliest_start = int(np.min(window_starts[candidates]))
            lead = max(0, min(s - earliest_start, pred_len_eff))
            lead_times_clean.append(int(lead))
        mean_lead_time_patch_clean = float(np.mean(lead_times_clean)) if lead_times_clean else None

        prognosis_errors_clean = []
        for i in range(len(pred_t)):
            if not (true_alarm_patch[i] and input_clean[i]):
                continue
            true_steps = np.where(true_patch_alarm[i])[0]
            pred_steps = np.where(pred_patch_alarm[i])[0]
            if true_steps.size > 0 and pred_steps.size > 0:
                prognosis_errors_clean.append(abs(int(pred_steps[0]) - int(true_steps[0])))
        mean_prognosis_error_clean = float(np.mean(prognosis_errors_clean)) if prognosis_errors_clean else None

        # --- Quality-filtered variants ---
        ratio_pred_in_true_qf = safe_mean(
            pred_alarm_last5_qf.astype(float), true_alarm_last5)
        ratio_pred_in_no_true_qf = safe_mean(
            pred_alarm_last5_qf.astype(float), ~true_alarm_patch)

        lead_times_qf = []
        for (s, _) in true_events:
            candidates = np.where(
                pred_alarm_patch_qf
                & (window_starts <= s) & (s <= (window_starts + pred_len_eff - 1))
            )[0]
            if candidates.size == 0:
                lead_times_qf.append(0)
                continue
            earliest_start = int(np.min(window_starts[candidates]))
            lead = max(0, min(s - earliest_start, pred_len_eff))
            lead_times_qf.append(int(lead))
        mean_lead_time_patch_qf = float(np.mean(lead_times_qf)) if lead_times_qf else None

        prognosis_errors_qf = []
        for i in range(len(pred_t)):
            if not (true_alarm_patch[i] and pred_quality_ok[i]):
                continue
            true_steps = np.where(true_patch_alarm[i])[0]
            pred_steps = np.where(pred_patch_alarm[i])[0]
            if true_steps.size > 0 and pred_steps.size > 0:
                prognosis_errors_qf.append(abs(int(pred_steps[0]) - int(true_steps[0])))
        mean_prognosis_error_qf = float(np.mean(prognosis_errors_qf)) if prognosis_errors_qf else None

        # --- Clean + quality-filtered ---
        pred_alarm_patch_clean_qf = pred_alarm_patch & input_clean & pred_quality_ok
        pred_alarm_last5_clean_qf = pred_alarm_last5 & input_clean & pred_quality_ok

        ratio_pred_in_true_clean_qf = safe_mean(
            pred_alarm_last5_clean_qf.astype(float), true_alarm_last5_clean)
        ratio_pred_in_no_true_clean_qf = safe_mean(
            pred_alarm_last5_clean_qf.astype(float),
            ~true_alarm_patch_clean & input_clean)

        lead_times_clean_qf = []
        for (s, _) in true_events:
            candidates = np.where(
                pred_alarm_patch_clean_qf
                & (window_starts <= s) & (s <= (window_starts + pred_len_eff - 1))
            )[0]
            if candidates.size == 0:
                lead_times_clean_qf.append(0)
                continue
            earliest_start = int(np.min(window_starts[candidates]))
            lead = max(0, min(s - earliest_start, pred_len_eff))
            lead_times_clean_qf.append(int(lead))
        mean_lead_time_patch_clean_qf = float(np.mean(lead_times_clean_qf)) if lead_times_clean_qf else None

        prognosis_errors_clean_qf = []
        for i in range(len(pred_t)):
            if not (true_alarm_patch[i] and input_clean[i] and pred_quality_ok[i]):
                continue
            true_steps = np.where(true_patch_alarm[i])[0]
            pred_steps = np.where(pred_patch_alarm[i])[0]
            if true_steps.size > 0 and pred_steps.size > 0:
                prognosis_errors_clean_qf.append(abs(int(pred_steps[0]) - int(true_steps[0])))
        mean_prognosis_error_clean_qf = float(np.mean(prognosis_errors_clean_qf)) if prognosis_errors_clean_qf else None

        # --- Assemble metrics_C ---
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

        # Save per-model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / 'metrics_C.json').write_text(json.dumps(metrics_C, indent=2))
        metrics_all_C[model_name] = metrics_C
        print(f'[{model_name}] saved metrics_C to {result_dir}')

    # --- Plots ---
    target_name = str(target).split('_')[-1] if '_' in str(target) else str(target)
    fig_suffix = str(args.figure_suffix or '').strip()
    fig_dir = Path('./figures') / f'{target_name}{fig_suffix}'
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_keys_c = [
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

    plot_keys_c_radar = [
        'mse_true_alarm_patch', 'mse_no_true_alarm_patch',
        'mse_pred_alarm_patch', 'mse_no_pred_alarm_patch',
        'ratio_pred_in_true_alarm_patches_clean_qf',
        'ratio_pred_in_no_true_alarm_patches_clean_qf',
        'mean_lead_time_patch_clean_qf', 'mean_prognosis_error_clean_qf',
    ]

    plot_metrics(metrics_all_C, 'C', plot_keys_c, fig_dir)
    plot_radar(metrics_all_C, plot_keys_c_radar, 'C', fig_dir)

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
        'n_pred_alarm_patches', 'n_pred_alarm_patches_qf', 'n_quality_rejected',
        'rmse_true_alarm_patch', 'rmse_no_true_alarm_patch',
        'rmse_pred_alarm_patch', 'rmse_no_pred_alarm_patch', 'rmse_all_patches',
        'mae_true_alarm_patch', 'mae_no_true_alarm_patch',
        'mae_pred_alarm_patch', 'mae_no_pred_alarm_patch', 'mae_all_patches',
        'tp_window', 'fp_window', 'tn_window', 'fn_window',
        'tp_window_clean', 'fp_window_clean', 'tn_window_clean', 'fn_window_clean',
        'tp_window_qf', 'fp_window_qf', 'tn_window_qf', 'fn_window_qf',
        'tp_window_clean_qf', 'fp_window_clean_qf', 'tn_window_clean_qf', 'fn_window_clean_qf',
    ]
    summary_rows = []
    for entry in model_dirs:
        mc = metrics_all_C.get(entry['name'])
        if not mc:
            continue
        row = {'model': entry['name']}
        for k in summary_cols:
            row[k] = mc.get(k, None)
        summary_rows.append(row)

    suffix = str(args.summary_suffix or '').strip()
    summary_name = f'{target_name}_Summary{suffix}'
    summary_dir = Path('./results') / summary_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows, columns=['model'] + summary_cols)
    summary_csv = summary_dir / 'summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    try:
        summary_md = summary_dir / 'summary.md'
        summary_md.write_text(summary_df.to_markdown(index=False), encoding='utf-8')
    except Exception:
        pass
    print(f'saved summary: {summary_csv}')


if __name__ == '__main__':
    main()
