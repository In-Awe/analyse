import os
import argparse
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
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
        time_col_candidates = ['timestamp', 'time', 'date', 'datetime', 'ts', 'datetime_utc']
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

def detect_timestamp_gaps(df):
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min', tz=df.index.tz)
    missing = full_idx.difference(df.index)
    if missing.empty:
        return pd.DataFrame()
    s = pd.Series(1, index=missing)
    groups = (s.index.to_series().diff() != pd.Timedelta('1min')).cumsum()
    gaps = []
    for _, g in s.groupby(groups):
        gaps.append({'gap_start': g.index[0], 'gap_end': g.index[-1], 'gap_minutes': len(g)})
    return pd.DataFrame(gaps)

def add_basic_features(df):
    df['log_return'] = np.log(df['close']).diff()
    df['return_sq'] = df['log_return'] ** 2
    df['range'] = df['high'] - df['low']
    df['tick_imbalance'] = (df['close'] - df['open']) / (df['range'] + 1e-12)
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['vwap_cum'] = (tp * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-12)
    return df

def advanced_spike_detection(df, window=60, ret_mad_thresh=8.0, vol_surge_thresh=6.0):
    df_res = df.copy()
    if 'log_return' not in df_res.columns:
        df_res = add_basic_features(df_res)
    roll_mad_ret = df_res['log_return'].rolling(window).apply(lambda x: robust_mad(x, scale='normal'), raw=False)
    ret_z = (df_res['log_return'] / roll_mad_ret.replace(0, np.nan).ffill().bfill()).abs()
    roll_med_vol = df_res['volume'].rolling(window).median()
    vol_surge = df_res['volume'] / roll_med_vol.replace(0, np.nan).ffill().bfill()
    spike_mask = (ret_z > ret_mad_thresh) & (vol_surge > vol_surge_thresh)
    return df_res[spike_mask]

def main(args):
    print(f"Loading and auditing {args.input}...")
    df = load_data(args.input)
    gaps = detect_timestamp_gaps(df)
    spikes = advanced_spike_detection(df)
    ensure_dir(args.output)
    gaps.to_csv(os.path.join(args.output, 'gaps_report.csv'))
    spikes.to_csv(os.path.join(args.output, 'anomalies_report.csv'))
    print(f"Data audit complete. Reports saved in '{args.output}'.")
    print(f"- Found {len(gaps)} timestamp gaps.")
    print(f"- Detected {len(spikes)} potential spike anomalies.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1 Data Audit Pipeline")
    parser.add_argument('--input', required=True, help="Path to the input CSV data.")
    parser.add_argument('--output', default='reports', help="Directory to save audit reports.")
    args = parser.parse_args()
    main(args)
