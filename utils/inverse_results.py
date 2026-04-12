"""
inverse_results.py — Inverse-transform (denormalize) all pred.npy / true.npy
in TSFM results directories.

Fits StandardScaler on the training CSV (same as MultivariateDatasetYAMLSplit),
then applies inverse_transform to each model's saved results in-place.

Usage:
    cd /home/aicode/sherwin/TSFM
    python utils/inverse_results.py --config setting/batch_metrics_pca101a.yaml
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def load_csv_drop_time(path):
    df = pd.read_csv(path)
    if len(df) > 0 and isinstance(df[df.columns[0]].iloc[0], str):
        df = df[df.columns[1:]]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    params = cfg.get('params', {}) or {}

    target = params['target']
    data_root = Path(params.get('data_root', '/home/aicode/sherwin/dataset/JJ'))
    results_root = Path(params.get('results_root', './results'))
    split_file = params.get('split_file', 'setting/PCA101A.yaml')

    # Read train files from the TSFM split YAML
    split_cfg = yaml.safe_load(Path(split_file).read_text()) or {}
    train_files = split_cfg.get('train', [])

    # --- Fit two scalers: all-columns (6D) and target-only (1D) ---
    train_all = []
    train_target = []
    for f in train_files:
        df = load_csv_drop_time(data_root / f)
        train_all.append(df.values.astype(np.float32))
        train_target.append(df[[target]].values.astype(np.float32))

    scaler_all = StandardScaler()
    scaler_all.fit(np.concatenate(train_all, axis=0))

    scaler_target = StandardScaler()
    scaler_target.fit(np.concatenate(train_target, axis=0))

    print(f"Scaler (all {scaler_all.n_features_in_} cols): mean={scaler_all.mean_}, scale={scaler_all.scale_}")
    print(f"Scaler (target only): mean={scaler_target.mean_}, scale={scaler_target.scale_}")

    # Target column index in the all-columns scaler
    target_mean = scaler_all.mean_[-1]
    target_scale = scaler_all.scale_[-1]

    model_dirs = cfg.get('model_dirs', [])

    for entry in model_dirs:
        name = entry['name']
        result_dir = results_root / entry['result_dir']
        pred_path = result_dir / 'pred.npy'
        true_path = result_dir / 'true.npy'

        if not pred_path.exists():
            print(f'[{name}] SKIP — pred.npy not found')
            continue

        pred = np.load(pred_path)
        true = np.load(true_path)

        print(f'[{name}] pred={pred.shape}  true={true.shape}', end='')

        if pred.ndim == 3 and pred.shape[-1] == 6:
            # M mode: (N, 96, 6) — use all-columns scaler
            N, T, C = pred.shape
            pred = scaler_all.inverse_transform(pred.reshape(-1, C)).reshape(N, T, C)
            true = scaler_all.inverse_transform(true.reshape(-1, C)).reshape(N, T, C)

        elif pred.ndim == 3 and pred.shape[-1] == 1:
            # S/U mode: (N, 96, 1) — use target-only scaler
            N, T, C = pred.shape
            pred = scaler_target.inverse_transform(pred.reshape(-1, C)).reshape(N, T, C)
            true = scaler_target.inverse_transform(true.reshape(-1, C)).reshape(N, T, C)

        elif pred.ndim == 2:
            # Co mode (squeezed): (N, 96) — use target col from all-columns scaler
            pred = pred * target_scale + target_mean
            true = true * target_scale + target_mean

        else:
            print(f'  SKIP — unexpected shape')
            continue

        np.save(pred_path, pred)
        np.save(true_path, true)
        print(f'  -> saved  pred_range=[{pred.min():.1f}, {pred.max():.1f}]')


if __name__ == '__main__':
    main()
