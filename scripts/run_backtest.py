#!/usr/bin/env python3
"""Command-line runner for Phase IV backtests (vectorized).
Usage:
  python scripts/run_backtest.py --data data/cleaned/BTCUSD_1min.cleaned.csv --out artifacts/backtest/run1
"""
from pathlib import Path
import argparse
import yaml
import pandas as pd

from src.backtest.backtester import Backtester, BacktestConfig

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--out", default="artifacts/backtest/default")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = BacktestConfig(**cfg_dict.get("backtest", {}))

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    # require 'signal' column; simple default: SIDEWAYS if missing
    if "signal" not in df.columns:
        df["signal"] = 0
    # ensure atr exists
    if "atr" not in df.columns:
        df["returns"] = (df["close"].astype(float).pct_change()).fillna(0)
        df["atr"] = df["returns"].rolling(14).std().fillna(0) * df["close"]

    bt = Backtester(df, cfg)
    equity, trades = bt.run()
    bt.save_outputs(args.out)
    print("Backtest complete. Outputs saved to", args.out)

if __name__ == "__main__":
    main()
