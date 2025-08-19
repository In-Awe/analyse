#!/usr/bin/env python3
"""
Generate a small synthetic BTCUSD 1-minute OHLCV CSV for smoke/backtest CI.
Produces: data/cleaned/BTCUSD_1min.cleaned.csv (creates dirs if needed)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import datetime as dt

def generate(start_ts=None, minutes=24*60, out="data/cleaned/BTCUSD_1min.cleaned.csv"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if start_ts is None:
        start = pd.Timestamp.utcnow().floor('T') - pd.Timedelta(minutes=minutes)
    else:
        start = pd.Timestamp(start_ts)
    idx = pd.date_range(start=start, periods=minutes, freq='T', tz=None)
    # simple geometric brownian motion for price
    np.random.seed(42)
    mu = 0.0
    sigma = 0.0008  # ~0.08% per minute
    ret = np.random.normal(loc=mu, scale=sigma, size=minutes)
    price = 30000 * np.exp(np.cumsum(ret))  # arbitrary base price
    # build OHLC around price with tiny spreads
    o = price * (1 + np.random.normal(0, 0.0001, size=minutes))
    c = price
    h = np.maximum(o, c) * (1 + np.abs(np.random.normal(0, 0.0005, size=minutes)))
    l = np.minimum(o, c) * (1 - np.abs(np.random.normal(0, 0.0005, size=minutes)))
    vol = np.abs(np.random.normal(50, 20, size=minutes))
    df = pd.DataFrame({
        "timestamp": idx,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol
    })
    df.to_csv(out, index=False)
    print(f"Generated sample data -> {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/cleaned/BTCUSD_1min.cleaned.csv")
    p.add_argument("--minutes", type=int, default=24*60)
    p.add_argument("--start", default=None)
    args = p.parse_args()
    generate(start_ts=args.start, minutes=args.minutes, out=args.out)
