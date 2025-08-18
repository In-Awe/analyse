"""
Data ingestion + integrity audit for BTCUSDT_1m_December_2024.csv

This script performs the initial data quality checks as specified in Phase I 
of the HFT system development directive. It addresses:
1.  Duplicate timestamp detection and removal.
2.  Identification of missing timestamps (gaps) in the 1-minute series.
3.  Imputation of missing data using a forward-fill/backward-fill strategy.
4.  Anomalous spike detection using a rolling Z-score.
5.  Generation of two new features: `is_imputed_datapoint` and `spike_magnitude`.
6.  A preliminary statistical analysis of log returns, including an ADF test.

Outputs:
 - data/clean/BTCUSDT_1m_December_2024.cleaned.csv
 - data/reports/BTCUSDT_1m_December_2024_audit.json
"""
import os
import json
from datetime import timedelta
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# --- CONFIGURATION ---
# Adjusted paths and column names to match the provided dataset.
RAW_CSV = "BTCUSDT_1m_December_2024.csv"
CLEAN_CSV = "data/clean/BTCUSDT_1m_December_2024.cleaned.csv"
REPORT_JSON = "data/reports/BTCUSDT_1m_December_2024_audit.json"
TIMESTAMP_COL = "datetime_utc" # MODIFIED: Column name in the source CSV is 'datetime_utc'
TIMEZONE = "UTC"
FREQ = "1T"  # pandas frequency string for 1 minute
SPIKE_ZSCORE_THRESHOLD = 8.0  # Z-score threshold for flagging extreme price spikes

# --- SCRIPT EXECUTION ---

def setup_directories():
    """Create output directories if they don't exist."""
    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)

def read_and_prepare_data(path):
    """
    Reads the raw CSV, standardizes the timestamp column, and sorts the data.
    """
    print(f"Reading raw data from {path}...")
    df = pd.read_csv(path)
    
    # Coerce the timestamp column to datetime objects with UTC timezone.
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    
    # Sort by timestamp to ensure correct chronological order.
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    print(f"Successfully read {len(df)} rows.")
    return df

def audit_and_clean(df):
    """
    Performs the main data audit and cleaning pipeline.
    """
    print("Starting data audit and cleaning process...")
    report = {}

    # 1. Handle Duplicates
    report['rows_raw'] = int(len(df))
    dup_mask = df.duplicated(subset=[TIMESTAMP_COL], keep=False)
    report['duplicate_timestamp_count'] = int(dup_mask.sum())
    if report['duplicate_timestamp_count'] > 0:
        # If duplicates exist, we keep the last recorded entry for that timestamp.
        df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep='last').reset_index(drop=True)
        print(f"Removed {report['duplicate_timestamp_count']} duplicate rows.")

    # 2. Identify and Fill Gaps
    df = df.set_index(TIMESTAMP_COL)
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FREQ, tz=TIMEZONE)
    
    report['first_timestamp'] = str(df.index.min())
    report['last_timestamp'] = str(df.index.max())
    report['expected_rows_for_timespan'] = int(len(full_idx))
    report['missing_timestamps_count'] = int(len(full_idx.difference(df.index)))

    # Reindex the dataframe to the full, gapless index and mark imputed points.
    df_reindexed = df.reindex(full_idx)
    df_reindexed['is_imputed'] = df_reindexed['close'].isna()
    
    # Use forward-fill then backward-fill to handle missing OHLCV data.
    # This is a common approach to preserve continuity.
    fill_cols = ['open', 'high', 'low', 'close', 'volume']
    df_ffill = df_reindexed.copy()
    df_ffill[fill_cols] = df_ffill[fill_cols].ffill().bfill()
    print(f"Filled {report['missing_timestamps_count']} missing timestamps.")

    # 3. Detect Spikes (Outliers)
    # We use a rolling z-score to detect anomalous price movements relative to recent history.
    rolling_window = 60 # 1 hour window
    roll_mean = df_ffill['close'].rolling(window=rolling_window, min_periods=10).mean()
    roll_std = df_ffill['close'].rolling(window=rolling_window, min_periods=10).std().replace(0, np.nan)
    zscore = (df_ffill['close'] - roll_mean) / roll_std
    
    spike_mask = (zscore.abs() > SPIKE_ZSCORE_THRESHOLD) & df_ffill['close'].notna()
    report['spike_candidates_found'] = int(spike_mask.sum())
    df_ffill['spike_zscore'] = zscore
    df_ffill['spike_candidate'] = spike_mask
    print(f"Identified {report['spike_candidates_found']} potential price spikes.")

    # 4. Feature Engineering (as per directive)
    df_ffill['is_imputed_datapoint'] = df_ffill['is_imputed'].astype(int)
    df_ffill['spike_magnitude'] = df_ffill['spike_zscore'].fillna(0).abs()

    # 5. Calculate Log Returns for Statistical Analysis
    df_ffill['log_return_1m'] = np.log(df_ffill['close'] / df_ffill['close'].shift(1))
    df_ffill['log_return_1m'] = df_ffill['log_return_1m'].fillna(0)

    # 6. Basic Distributional Statistics
    lr = df_ffill['log_return_1m'].replace([np.inf, -np.inf], np.nan).dropna()
    report['logreturn_mean'] = float(lr.mean())
    report['logreturn_std'] = float(lr.std())
    report['logreturn_skew'] = float(lr.skew())
    report['logreturn_kurtosis'] = float(lr.kurtosis())

    # 7. ADF Stationarity Test for Returns
    try:
        adf_res = adfuller(lr, autolag='AIC')
        report['adf_stationarity_test'] = {
            'statistic': float(adf_res[0]),
            'p_value': float(adf_res[1]),
            'is_stationary_at_5_percent': adf_res[1] < 0.05,
            'lags_used': int(adf_res[2])
        }
    except Exception as e:
        report['adf_stationarity_test'] = {'error': str(e)}

    # Finalize the cleaned dataframe
    cleaned_df = df_ffill.reset_index().rename(columns={'index': TIMESTAMP_COL})
    
    print("Audit and cleaning complete.")
    return cleaned_df, report

def main():
    """
    Main execution function.
    """
    if not os.path.exists(RAW_CSV):
        print(f"Error: Raw CSV not found at '{RAW_CSV}'.")
        print("Please ensure the file is available in the correct path.")
        return

    setup_directories()
    raw_df = read_and_prepare_data(RAW_CSV)
    cleaned_df, audit_report = audit_and_clean(raw_df)
    
    # Save the cleaned data
    cleaned_df.to_csv(CLEAN_CSV, index=False)
    print(f"Cleaned data written to: {CLEAN_CSV}")

    # Save the audit report
    with open(REPORT_JSON, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"Audit report written to: {REPORT_JSON}")
    print("\n--- Audit Summary ---")
    print(json.dumps(audit_report, indent=2))
    print("---------------------\n")


if __name__ == "__main__":
    main()