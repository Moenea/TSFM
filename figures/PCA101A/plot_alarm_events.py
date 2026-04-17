"""
plot_alarm_events.py — Per-event visualization of the 10 alarm segments
on the PCA101A test series, with 6-model predictions overlaid.

Alarm logic mirrors utils/batch_metrics.py:
    alarm = (true_series > high) | (true_series < low)
then groups contiguous True runs into events.

For each event at original-series start index s, we overlay the 15-step
forecast issued 5 steps before the alarm starts, i.e. the window whose
first forecast step lands at original index (s - 5).  The patch index is

    patch_index = (s - 5) - seq_len           # seq_len = 1440

so the forecast covers original indices [s-5, s+9].  Only the first 15
steps (= 15 minutes) are plotted.

All 10 events on this test series are LOW alarms (< 205), so only the
low threshold line is drawn.

Outputs (saved next to this script):
    overview_all_events.png
    event_00_start{S}_end{E}_patch{P}.png   ... event_09_*.png
    alarm_events_PCA101A.csv

Run:
    python TSFM/figures/PCA101A/plot_alarm_events.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CM = 1 / 2.54  # cm -> inch
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size':   7.5,
    'axes.titlesize':  7.5,
    'axes.labelsize':  7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
})

# --- config ---------------------------------------------------------------
TARGET    = '0202B_PCA101A'
LIMIT_CSV = Path('/home/aicode/sherwin/dataset/JJ/2-limit-1.csv')
TEST_CSV  = Path('/home/aicode/sherwin/dataset/JJ/PCA101A/'
                 '2FCC_PCA101A_smooth_testing1_long.csv')
RESULTS   = Path('/home/aicode/sherwin/TSFM/results')
OUT_DIR   = Path(__file__).resolve().parent

SEQ_LEN     = 1440
PRED_LEN    = 96
EVAL_STEPS  = 15      # only plot first 15 steps of each forecast
LEAD_STEPS  = 10       # forecast issued 5 steps (= 5 min) before alarm starts
LOOKBACK    = 60+LEAD_STEPS     # samples shown before alarm start
LOOKFORWARD = 15-LEAD_STEPS     # samples shown after alarm start  (window = 420 wide)

# --- MyTimeXer baseline settings -----------------------------------------
MTX_RESULTS  = Path('/home/aicode/sherwin/MyTimeXer/results')
MTX_SEQ_LEN  = 30
MTX_PRED_LEN = 15
# Number of forecast windows a "full" MyTimeXer model produces on this test
# series.  BiLSTM produces fewer (starts later) and is handled dynamically.
MTX_N_FULL   = 194956

MODELS = [
    # Zeroshot family — blues
    ('Zeroshot-U',  'zeroshot_PCA101A_U',  '#08519c'),
    ('Zeroshot-Co', 'zeroshot_PCA101A_Co', '#6baed6'),
    # Finetuned family — greens
    ('Finetuned-U', 'forecast_PCA101A_U_full_shot_timer_xl_'
                    'MultivariateDatasetYAMLSplit_sl1440_it96_ot96_'
                    'lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0',
                    '#006d2c'),
    ('Finetuned-Co','forecast_PCA101A_Co_full_shot_timer_xl_'
                    'MultivariateDatasetYAMLSplit_sl1440_it96_ot96_'
                    'lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0',
                    '#74c476'),
    # Partial15 family — oranges
    ('Partial15-U', 'forecast_PCA101A_U_partial15_timer_xl_'
                    'MultivariateDatasetYAMLSplit_sl1440_it96_ot96_'
                    'lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0',
                    '#a63603'),
    ('Partial15-Co','forecast_PCA101A_Co_partial15_timer_xl_'
                    'MultivariateDatasetYAMLSplit_sl1440_it96_ot96_'
                    'lr5e-06_bt32_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0',
                    '#fd8d3c'),
]

# --- MyTimeXer baselines ---------------------------------------------------
# tuple: (display_name, model_name_for_dirname, dm, el, df, des_suffix, color)
MTX_MODELS = [
    ('CNNLSTM',        'CNNLSTM',        230, 2, 1408, 'CNNLSTM-MS',        '#78909C'),
    # ('LSTMGRU',        'LSTMGRU',        316, 1, 2048, 'LSTMGRU-MS',        '#9E9E9E'),
    # ('STAConvBiLSTM',  'STAConvBiLSTM',  268, 1, 2048, 'STAConvBiLSTM-MS',  '#FFB300'),
    # ('TCNTransformer', 'TCNTransformer', 128, 2,  512, 'TCNTransformer-MS', '#66BB6A'),
    # ('DiPCALSTM',      'DiPCALSTM',      472, 2, 2048, 'DiPCALSTM-MS',      '#8D6E63'),
    ('TimeXer',        'TimeXer',        248, 1,  992, 'Timexer-MS',        '#5C6BC0'),
    ('ASProNet',       'GTProgerV13',    248, 1,  992, 'GTProgerV13-MS',    '#7CB342'),
]


def mtx_result_dir(model_name: str, dm: int, el: int, df: int, des: str) -> Path:
    name = ('long_term_forecast_JJ_PCA101A_MS_30_30_15_'
            f'{model_name}_custom_ftMS_sl30_ll30_pl15_dm{dm}_nh8_'
            f'el{el}_dl1_df{df}_expand2_dc4_fc3_ebtimeF_dtTrue_{des}_0_'
            f'dsJJ_PCA101A_t0202B_PCA101A')
    return MTX_RESULTS / name


def contiguous_events(mask):
    events, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(mask) - 1))
    return events


def load_pred_target(result_dir: Path) -> np.ndarray:
    """Load pred.npy and reduce to 2D (N, pred_len) on the target column."""
    arr = np.load(result_dir / 'pred.npy')
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    return arr


def main():
    df_limit = pd.read_csv(LIMIT_CSV, index_col=0)
    high = float(df_limit.loc[TARGET].iloc[1])
    low  = float(df_limit.loc[TARGET].iloc[2])

    arr  = pd.read_csv(TEST_CSV)[TARGET].to_numpy()
    n    = len(arr)
    mask = (arr > high) | (arr < low)
    events = contiguous_events(mask)

    n_windows_expected = max(0, n - SEQ_LEN - PRED_LEN + 1)
    print(f'target={TARGET}  high={high}  low={low}  '
          f'series_len={n}  events={len(events)}  '
          f'expected_windows={n_windows_expected}')

    # load TSFM predictions
    preds = {}
    for name, dirname, _ in MODELS:
        p = load_pred_target(RESULTS / dirname)
        if p.shape[0] != n_windows_expected:
            print(f'  WARN {name}: pred has {p.shape[0]} windows, '
                  f'expected {n_windows_expected}')
        preds[name] = p

    # load MyTimeXer predictions; record the original-series offset per model.
    # A MyTimeXer window at raw index i covers original steps [i+MTX_SEQ_LEN,
    # i+MTX_SEQ_LEN+MTX_PRED_LEN-1].  Full models start at i=0 (origin offset
    # MTX_SEQ_LEN=30).  BiLSTM has fewer windows — its i=0 corresponds to a
    # later original step; we derive that shift from the window count.
    mtx_preds = {}       # name -> ndarray (N_raw, MTX_PRED_LEN)
    mtx_origin = {}      # name -> origin (original index of raw i=0 forecast)
    for name, m_name, dm, el, df, des, _color in MTX_MODELS:
        d = mtx_result_dir(m_name, dm, el, df, des)
        p = np.load(d / 'pred.npy')
        if p.ndim == 3:
            p = p[:, :, -1]
        mtx_preds[name] = p
        # shift of raw[0] relative to a "full" MyTimeXer run's raw[0]
        row_shift = MTX_N_FULL - p.shape[0]
        mtx_origin[name] = MTX_SEQ_LEN + row_shift
        if row_shift != 0:
            print(f'  NOTE {name}: {p.shape[0]} windows, '
                  f'origin shifted by {row_shift} (raw[0] forecast at '
                  f'original index {mtx_origin[name]})')

    # refresh event table (with patch index)
    rows = []
    for s, e in events:
        patch = s - LEAD_STEPS - SEQ_LEN
        rows.append((s, e, e - s + 1,
                     'HIGH' if arr[s] > high else 'LOW',
                     float(arr[s]),
                     float(arr[s:e + 1].min()),
                     float(arr[s:e + 1].max()),
                     patch))
    pd.DataFrame(rows, columns=[
        'start', 'end', 'length', 'type',
        'first_value', 'min_value', 'max_value', 'patch_index_lead5',
    ]).to_csv(OUT_DIR / 'alarm_events_PCA101A.csv', index=False)

    # --- overview figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8*CM, 6*CM))
    ax.plot(arr, color='#293890', linewidth=0.4, label='value')
    ax.axhline(low, color='orange', linewidth=0.8, linestyle='--',
               label=f'low={low}')
    for i, (s, e) in enumerate(events):
        ax.axvspan(s, e + 1, color='red', alpha=0.25)
        ax.text((s + e) / 2, ax.get_ylim()[1], f'{i}',
                ha='center', va='top', fontsize=8, color='red')
    ax.set_title(f'{TARGET} — all {len(events)} alarm events on test series')
    ax.set_xlabel('sample index'); ax.set_ylabel('value')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'overview_all_events.png', dpi=180)
    plt.close(fig)
    print('saved overview_all_events.png')

    # --- per-event zoom plots --------------------------------------------
    for i, (s, e) in enumerate(events):
        patch = s - LEAD_STEPS - SEQ_LEN
        if patch < 0 or patch >= n_windows_expected:
            print(f'  event {i}: patch {patch} out of range, skipping forecasts')
            valid_patch = False
        else:
            valid_patch = True

        # forecast horizon on original-index axis: [s-LEAD, s-LEAD+EVAL_STEPS-1]
        fc_start = s - LEAD_STEPS
        fc_x = np.arange(fc_start, fc_start + EVAL_STEPS)

        x0 = max(0, s - LOOKBACK)
        x1 = min(n, s + LOOKFORWARD + 1)
        x  = np.arange(x0, x1)
        y  = arr[x0:x1]
        # right edge of the alarm shading (clipped to window)
        seg_end = min(e, x1 - 1)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(12 * CM, 12 * CM), sharex=True,
            constrained_layout=True)

        # shared layers (true / low / forecast issue / alarm shading) ----
        def draw_common(ax):
            ax.plot(x, y, color='#293890', linewidth=1.8,
                    label='True', zorder=5)
            ax.axhline(low, color='orange', linewidth=0.6, linestyle='--',
                       label=f'low={low}')
            ax.axvspan(s, seg_end + 1, color='red', alpha=0.10)
            ax.axvline(s, color='red',  linewidth=0.6, linestyle=':')
            if e <= x1 - 1:
                ax.axvline(e, color='gray', linewidth=0.6, linestyle=':')
            ax.axvline(fc_start, color='black', linewidth=0.6,
                       linestyle='-.', label=f'forecast issue t={fc_start}')

        draw_common(ax_top)
        draw_common(ax_bot)

        # --- top panel: TSFM (Timer-XL based) ---------------------------
        if valid_patch:
            for name, _, color in MODELS:
                yhat = preds[name][patch, :EVAL_STEPS]
                ax_top.plot(fc_x, yhat, color=color, linewidth=1.2,
                            alpha=0.9, label=name, zorder=2)

        # --- bottom panel: MyTimeXer baselines (dashed) -----------------
        # MyTimeXer raw index i: forecast starts at original index
        # mtx_origin[name] + i.  Solve for i given fc_start.
        for name, *_rest, color in MTX_MODELS:
            raw_len = mtx_preds[name].shape[0]
            mtx_i = fc_start - mtx_origin[name]
            if mtx_i < 0 or mtx_i >= raw_len:
                print(f'  event {i}: {name} raw idx {mtx_i} out of range '
                      f'[0, {raw_len}), skipping')
                continue
            yhat = mtx_preds[name][mtx_i, :EVAL_STEPS]
            ax_bot.plot(fc_x, yhat, color=color, linewidth=1.2,
                        linestyle='--', alpha=0.9, label=name, zorder=2)

        ax_top.set_title(
            f'event {i:02d}  |  start={s}  end={e}  len={e - s + 1}  '
            f'type={"HIGH" if arr[s] > high else "LOW"}  '
            f'patch_index={patch}  '
            f'(forecast covers [{fc_start}, {fc_start + EVAL_STEPS - 1}])')
        ax_top.set_ylabel('value  (Timer-XL based)')
        ax_bot.set_ylabel('value  (MyTimeXer baselines)')
        ax_bot.set_xlabel(
            f'sample index  (lookback {LOOKBACK} / '
            f'lookforward {LOOKFORWARD})')
        ax_top.legend(loc='best', ncol=2)
        ax_bot.legend(loc='best', ncol=2)

        out = OUT_DIR / f'event_{i:02d}_start{s}_end{e}_patch{patch}.png'
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f'saved {out.name}')


if __name__ == '__main__':
    main()
