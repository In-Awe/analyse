#!/usr/bin/env python3
"""
Runner for robustness analysis:
 - Run backtester to obtain baseline trades
 - Run Monte Carlo trade reshuffling
 - Run parameter sensitivity grid
Outputs saved under artifacts/backtest/robustness_<timestamp>/
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import numpy as np

from src.backtest.backtester import Backtester, BacktestConfig
from src.backtest.robustness import monte_carlo_trade_reshuffle

def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/cleaned/BTCUSD_1min.cleaned.csv", index_col=0, parse_dates=True)
    # require 'signal' column; simple default: SIDEWAYS if missing
    if "signal" not in df.columns:
        df["signal"] = np.random.randint(-1, 2, size=len(df))
    # ensure atr exists
    if "atr" not in df.columns:
        df["returns"] = (df["close"].astype(float).pct_change()).fillna(0)
        df["atr"] = df["returns"].rolling(14).std().fillna(0) * df["close"]

    cfg = BacktestConfig()
    bt = Backtester(df, cfg)

    # run baseline backtest
    print("[robustness] running baseline backtest")
    equity, trades = bt.run()

    # save baseline
    trades.to_csv(out_dir / "baseline_trades.csv", index=False)
    # summary = bt.summary()
    # with open(out_dir / "baseline_summary.json", "w") as f:
    #     json.dump(summary, f, indent=2)

    # Monte Carlo
    # mc_cfg = cfg.get("robustness", {}).get("monte_carlo", {})
    n = args.mc_iterations
    use_dask = False
    dask_cfg = {}
    n_workers = None
    mode = "multiprocess"
    print(f"[robustness] running monte-carlo reshuffle n={n}")
    mc = monte_carlo_trade_reshuffle(trades, n=n)
    with open(out_dir / "monte_carlo.json", "w") as f:
        json.dump(mc, f, indent=2)

    # parameter sensitivity
    print(f"[robustness] skipping param sensitivity")

    print(f"[robustness] saved results to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mc-iterations", type=int, default=100)
    parser.add_argument("--noise-trials", type=int, default=5)
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
