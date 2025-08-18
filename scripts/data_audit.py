import pandas as pd
import numpy as np
import json

def audit_data(input_path: str, output_path: str, k_spike: float = 8.0):
    """
    Audits the financial time series data.

    Args:
        input_path (str): Path to the raw data CSV file.
        output_path (str): Path to save the audit summary JSON.
        k_spike (float): Multiplier for spike detection.
    """
    # Load data
    df = pd.read_csv(input_path)

    # --- Timestamp audit ---
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df = df.set_index('datetime_utc')

    # Verify timezone-corrected timestamps (assuming UTC)
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')

    # Check for monotonic timestamps
    is_monotonic = df.index.is_monotonic_increasing

    # Check for 1-minute frequency and find missing timestamps
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min')
    missing_ts_count = len(full_range.difference(df.index))

    # --- Duplicates audit ---
    duplicates_count = df.duplicated().sum()

    # --- Zero volume during high volatility ---
    df['volume_30m_std'] = df['volume'].rolling(window=30).std()
    median_vol_std = df['volume_30m_std'].median()
    high_vol_windows = df[df['volume_30m_std'] > 2 * median_vol_std]
    zero_volume_in_high_vol_count = high_vol_windows[high_vol_windows['volume'] == 0].shape[0]

    # --- Price spike detection ---
    df['price_change'] = df['close'].diff().abs()
    df['rolling_std_30'] = df['price_change'].rolling(window=30).std()
    spikes = df[df['price_change'] > k_spike * df['rolling_std_30']]
    spike_count = spikes.shape[0]

    # --- Create summary report ---
    audit_summary = {
        'is_monotonic': bool(is_monotonic),
        'missing_ts': int(missing_ts_count),
        'duplicates': int(duplicates_count),
        'zero_volume_in_high_vol': int(zero_volume_in_high_vol_count),
        'spikes': int(spike_count)
    }

    # Save summary
    with open(output_path, 'w') as f:
        json.dump(audit_summary, f, indent=4)

    print(f"Audit summary saved to {output_path}")

if __name__ == '__main__':
    # This allows the script to be run from the command line
    import argparse
    parser = argparse.ArgumentParser(description='Data audit script.')
    parser.add_argument('--input', type=str, default='data/raw/BTCUSD_1min.csv', help='Input CSV file path.')
    parser.add_argument('--output', type=str, default='artifacts/audit_summary.json', help='Output JSON summary file path.')
    parser.add_argument('--k', type=float, default=8.0, help='Spike detection multiplier.')
    args = parser.parse_args()

    audit_data(args.input, args.output, args.k)
