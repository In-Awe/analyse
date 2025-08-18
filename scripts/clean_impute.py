import pandas as pd
import numpy as np
import json

def clean_impute_data(input_path: str, output_csv_path: str, output_log_path: str, k_moderate_spike: float = 8.0, k_severe_spike: float = 12.0):
    """
    Cleans and imputes financial time series data.

    Args:
        input_path (str): Path to the raw data CSV file.
        output_csv_path (str): Path to save the cleaned data CSV.
        output_log_path (str): Path to save the imputation log.
        k_moderate_spike (float): Multiplier for moderate spike detection.
        k_severe_spike (float): Multiplier for severe spike detection.
    """
    df = pd.read_csv(input_path)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df = df.set_index('datetime_utc')
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')

    imputation_log = []

    # --- Handle missing timestamps and impute ---
    df['is_imputed'] = False
    original_index = df.index
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min')
    df = df.reindex(full_range)

    missing_indices = df.index.difference(original_index)
    if not missing_indices.empty:
        df.loc[missing_indices, 'is_imputed'] = True
        imputation_log.append({'timestamp': missing_indices.to_list(), 'field': 'all', 'imputation_type': 'row_added'})

        # Impute prices with ffill
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        imputation_log.append({'timestamp': missing_indices.to_list(), 'field': price_cols, 'imputation_type': 'ffill'})

        # Impute volume with linear interpolation for short gaps
        df['volume'] = df['volume'].interpolate(method='linear', limit=5)
        imputation_log.append({'timestamp': missing_indices.to_list(), 'field': 'volume', 'imputation_type': 'linear_interpolation_short_gap'})

    # Identify long gaps (> 60 minutes)
    df['drop_period'] = df['close'].isnull().rolling(window=60, min_periods=1).max().astype(bool)


    # --- Spike Handling ---
    df['is_spike'] = False
    df['spike_magnitude'] = np.nan
    df['price_change'] = df['close'].diff().abs()
    df['rolling_std_30'] = df['price_change'].rolling(window=30).std()

    # Severe spikes
    severe_spikes_mask = df['price_change'] > k_severe_spike * df['rolling_std_30']
    df.loc[severe_spikes_mask, 'is_spike'] = True
    df.loc[severe_spikes_mask, 'spike_magnitude'] = df['price_change'] / df['rolling_std_30']

    # Moderate spikes
    moderate_spikes_mask = (df['price_change'] > k_moderate_spike * df['rolling_std_30']) & \
                           (df['price_change'] <= k_severe_spike * df['rolling_std_30'])
    df.loc[moderate_spikes_mask, 'is_spike'] = True
    df.loc[moderate_spikes_mask, 'spike_magnitude'] = df['price_change'] / df['rolling_std_30']

    # Fill NaNs for non-spike periods with 0
    df['spike_magnitude'] = df['spike_magnitude'].fillna(0.0)

    # Add smoothed columns for moderate spikes
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_smoothed'] = df[col]
        # Apply median smoothing for moderate spikes
        df.loc[moderate_spikes_mask, f'{col}_smoothed'] = df[col].rolling(window=3, center=True, min_periods=1).median()


    # Save cleaned data
    df.to_csv(output_csv_path)

    # Save imputation log
    if imputation_log:
        pd.DataFrame(imputation_log).to_csv(output_log_path, index=False)
    else:
        # Create empty log file if no imputations were made
        with open(output_log_path, 'w') as f:
            f.write("timestamp,field,imputation_type\n")

    print(f"Cleaned data saved to {output_csv_path}")
    print(f"Imputation log saved to {output_log_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data cleaning and imputation script.')
    parser.add_argument('--input', type=str, default='data/raw/BTCUSD_1min.csv', help='Input CSV file path.')
    parser.add_argument('--output-csv', type=str, default='data/cleaned/BTCUSD_1min.cleaned.csv', help='Output cleaned CSV file path.')
    parser.add_argument('--output-log', type=str, default='artifacts/imputation_log.csv', help='Output imputation log file path.')
    args = parser.parse_args()

    clean_impute_data(args.input, args.output_csv, args.output_log)
