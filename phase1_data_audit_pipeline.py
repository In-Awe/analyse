"""Phase I - Data Audit Pipeline (Enhanced) - MAD fallback patch + tests & CI
File: phase1_data_audit_pipeline.py
"""

import os
import argparse
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

def robust_mad(x, scale='normal'):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if scale == 'normal':
        return mad * 1.4826
    return mad

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_data(path, time_col_candidates=None):
    df = pd.read_csv(path)
    if time_col_candidates is None:
        time_col_candidates = ['timestamp', 'time', 'date', 'datetime', 'ts']

    found = None
    for c in time_col_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        found = df.columns[0]

    df.rename(columns={found: 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.set_index('timestamp').sort_index()

    name_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ['open', 'o']: name_map[col] = 'open'
        elif lc in ['high', 'h']: name_map[col] = 'high'
        elif lc in ['low', 'l']: name_map[col] = 'low'
        elif lc in ['close', 'c', 'price']: name_map[col] = 'close'
        elif 'volume' in lc or lc == 'v': name_map[col] = 'volume'
    df = df.rename(columns=name_map)
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        raise ValueError("Missing required OHLCV columns.")
    return df

def add_basic_features(df):
    df['log_return'] = np.log(df['close']).diff()
    df['return_sq'] = df['log_return'] ** 2
    return df

def advanced_spike_detection(df, window=60, ret_mad_thresh=8.0, vol_surge_thresh=6.0, min_volume=1.0):
    df_res = df.copy()
    if 'log_return' not in df_res.columns:
        df_res = add_basic_features(df_res)
    roll_mad_ret = df_res['log_return'].rolling(window).apply(lambda x: robust_mad(x, scale='normal'), raw=False)
    ret_z = (df_res['log_return'] / roll_mad_ret.replace(0, np.nan).ffill().bfill()).abs()
    roll_med_vol = df_res['volume'].rolling(window).median()
    vol_surge = df_res['volume'] / roll_med_vol.replace(0, np.nan).ffill().bfill()
    spike_mask = (ret_z > ret_mad_thresh) & (vol_surge > vol_surge_thresh) & (df_res['volume'] >= min_volume)
    return df_res[spike_mask]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Audit Pipeline")
    parser.add_argument('--input', required=True, help="Path to input CSV.")
    parser.add_argument('--output', default='reports', help="Directory for reports.")
    args = parser.parse_args()
    
    print(f"Auditing {args.input}...")
    df = load_data(args.input)
    spikes = advanced_spike_detection(df)
    ensure_dir(args.output)
    spikes.to_csv(os.path.join(args.output, 'anomalies_report.csv'))
    print(f"Audit complete. Found {len(spikes)} anomalies. Report saved in '{args.output}'.")
