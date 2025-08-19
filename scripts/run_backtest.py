#!/usr/bin/env python3
"""
Simple runner for the backtester skeleton. Supports --sample to run a small generated dataset
for CI smoke tests.
"""
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from src.backtest.backtester import VectorBacktester

def sample_df(n=300):
    # Build a tiny synthetic minute OHLCV series for smoke-run
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor('T'), periods=n, freq='T')
    price = 50000 + np.cumsum(np.random.normal(0, 1, size=n))
    df = pd.DataFrame({
        'timestamp': idx,
        'open': price,
        'high': price + np.abs(np.random.normal(0, 1, size=n)),
        'low': price - np.abs(np.random.normal(0, 1, size=n)),
        'close': price + np.random.normal(0, 0.2, size=n),
        'volume': np.abs(np.random.normal(1, 0.3, size=n)),
    }).set_index('timestamp')
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', help='Input cleaned CSV (OHLCV)', default=None)
    p.add_argument('--config', help='Config YAML', default='configs/trade_logic.yaml')
    p.add_argument('--outdir', help='Output directory', default='artifacts/backtest/run_local')
    p.add_argument('--sample', help='Run on synthetic sample data', action='store_true')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r') as fh:
        cfg = yaml.safe_load(fh)

    if args.sample or args.input is None:
        df = sample_df()
    else:
        df = pd.read_csv(args.input, parse_dates=['timestamp'], index_col='timestamp')

    # Minimal signals: use simple momentum: compare close to 5-min SMA
    df['sma5'] = df['close'].rolling(5, min_periods=1).mean()
    df['signal'] = np.where(df['close'] > df['sma5'], 'BUY', 'FLAT')

    # This will fail if src is not in python path. Let's add it.
    import sys
    sys.path.insert(0, os.getcwd())
    from src.backtest.backtester import VectorBacktester

    bt = VectorBacktester(df=df, config=cfg)
    trades, equity, summary = bt.run()

    trades.to_csv(outdir / 'trades.csv', index=False)
    equity.to_csv(outdir / 'equity_curve.csv', index=False)
    with open(outdir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("Backtest complete. Artifacts written to", outdir)

if __name__ == '__main__':
    main()
