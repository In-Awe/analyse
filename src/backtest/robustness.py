"""Robustness utilities:
 - monte_carlo trade reshuffle
 - noise-addition tests
 - parameter sweep harness skeleton
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def monte_carlo_trade_reshuffle(trades_df: pd.DataFrame, n: int = 1000, seed: int | None = None):
    """
    trades_df must have 'pnl' column (per trade P&L). Returns distribution dict.
    """
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df must include 'pnl' column")
    pnls = trades_df["pnl"].values
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        perm = rng.permutation(pnls)
        results.append(perm.sum())
    return {
        "n": n,
        "mean": float(np.mean(results)),
        "median": float(np.median(results)),
        "pct_5": float(np.percentile(results, 5)),
        "pct_95": float(np.percentile(results, 95)),
    }

def add_noise_and_backtest_prices(df: pd.DataFrame, noise_scale: float, trials: int, backtest_fn):
    """
    Adds gaussian noise to price series and runs 'backtest_fn' which accepts a df and returns a single scalar metric (e.g., final equity)
    """
    outputs = []
    returns_std = df["close"].pct_change().std()
    sigma = noise_scale * returns_std
    for _ in range(trials):
        noise = np.random.normal(loc=0.0, scale=sigma, size=len(df))
        df2 = df.copy()
        df2["close"] = df["close"] * (1 + noise)
        # keep OHLC consistent simply by shifting open/high/low to close (conservative)
        df2["open"] = df2["open"] * (1 + noise)
        df2["high"] = df2["high"] * (1 + noise)
        df2["low"] = df2["low"] * (1 + noise)
        out = backtest_fn(df2)
        outputs.append(out)
    arr = np.array(outputs)
    return {
        "trials": trials,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
