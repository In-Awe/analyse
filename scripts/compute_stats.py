import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

def compute_statistics(input_path: str, output_stats_path: str, output_plot_dir: str):
    """
    Computes preliminary statistics on the cleaned financial data.

    Args:
        input_path (str): Path to the cleaned data CSV.
        output_stats_path (str): Path to save the statistics JSON.
        output_plot_dir (str): Directory to save the generated plots.
    """
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index.name = 'datetime_utc'

    # --- Log Returns ---
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_returns'], inplace=True)

    # --- Basic Stats & Histogram ---
    log_returns = df['log_returns']
    skewness = skew(log_returns)
    kurt = kurtosis(log_returns)  # Fisher's kurtosis (normal is 0)

    plt.figure(figsize=(10, 6))
    sns.histplot(log_returns, bins=100, kde=True, stat='density', label='Log Returns')
    mu, std = norm.fit(log_returns)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Fit')
    plt.title('Histogram of 1-Min Log Returns')
    plt.legend()
    plt.savefig(f"{output_plot_dir}/log_returns_histogram.png")
    plt.close()

    # --- ADF Test ---
    adf_result = adfuller(log_returns)
    adf_pvalue = adf_result[1]

    # --- ACF Plots ---
    # ACF of returns
    plt.figure(figsize=(12, 6))
    plot_acf(log_returns, lags=240, title='ACF of Log Returns')
    plt.savefig(f"{output_plot_dir}/acf_returns.png")
    plt.close()

    # ACF of squared returns (volatility clustering)
    plt.figure(figsize=(12, 6))
    plot_acf(log_returns**2, lags=240, title='ACF of Squared Log Returns')
    plt.savefig(f"{output_plot_dir}/acf_squared_returns.png")
    plt.close()

    # --- Intraday Seasonality ---
    df['hour_of_day'] = df.index.hour
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute

    # Volatility by hour
    hourly_volatility = df.groupby('hour_of_day')['log_returns'].std()
    plt.figure(figsize=(10, 6))
    hourly_volatility.plot(kind='bar')
    plt.title('Average Volatility by Hour of Day')
    plt.ylabel('Std Dev of Log Returns')
    plt.savefig(f"{output_plot_dir}/hourly_volatility.png")
    plt.close()

    # Volume by hour
    hourly_volume = df.groupby('hour_of_day')['volume'].mean()
    plt.figure(figsize=(10, 6))
    hourly_volume.plot(kind='bar')
    plt.title('Average Volume by Hour of Day')
    plt.ylabel('Mean Volume')
    plt.savefig(f"{output_plot_dir}/hourly_volume.png")
    plt.close()

    # Heatmaps
    df['day_of_week'] = df.index.day_name()
    vol_heatmap_data = df.pivot_table(values='log_returns', index='day_of_week', columns='hour_of_day', aggfunc=np.std)
    plt.figure(figsize=(12, 7))
    sns.heatmap(vol_heatmap_data, cmap='viridis', annot=False)
    plt.title('Volatility (Std Dev of Log Returns) by Day and Hour')
    plt.savefig(f"{output_plot_dir}/volatility_heatmap.png")
    plt.close()

    # --- Save Statistics ---
    stats_summary = {
        'skewness': skewness,
        'kurtosis': kurt,
        'adf_pvalue': adf_pvalue,
        'notable_acf_lags': "Visual inspection of ACF plots needed"
    }

    with open(output_stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=4)

    print(f"Statistics saved to {output_stats_path}")
    print(f"Plots saved in {output_plot_dir}/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Statistical analysis script.')
    parser.add_argument('--input', type=str, default='data/cleaned/BTCUSD_1min.cleaned.csv', help='Input cleaned CSV file path.')
    parser.add_argument('--output-stats', type=str, default='artifacts/statistics.json', help='Output statistics JSON file path.')
    parser.add_argument('--output-plots', type=str, default='artifacts', help='Directory for saving plots.')
    args = parser.parse_args()

    import os
    if not os.path.exists(args.output_plots):
        os.makedirs(args.output_plots)

    compute_statistics(args.input, args.output_stats, args.output_plots)
