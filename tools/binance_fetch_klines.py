#!/usr/bin/env python3
"""
tools/binance_fetch_klines.py

Fetch historical 1m klines from Binance and save as CSV suitable for the repo's pipeline.
Requires: python-binance, pandas

Usage:
  pip install python-binance pandas
  export BINANCE_API_KEY=...
  export BINANCE_API_SECRET=...
  python tools/binance_fetch_klines.py --symbol BTCUSDT --year 2025 --month 7 --out data/raw
"""
import argparse
import os
from binance.client import Client
import pandas as pd
from datetime import datetime, timezone

def fetch_month(symbol="BTCUSDT", year=2024, month=6, interval="1m", out_dir="data/raw"):
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    client = Client(api_key=api_key, api_secret=api_secret)
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month+1, 1, tzinfo=timezone.utc)
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fetching {symbol} {interval} from {start_str} to {end_str} (UTC)")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open","high","low","close","volume","close_time",
        "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    os.makedirs(out_dir, exist_ok=True)
    outname = f"{symbol}_{interval}_{year:04d}-{month:02d}.csv"
    outpath = os.path.join(out_dir, outname)
    df.to_csv(outpath, index=False)
    print("Saved", outpath)
    return outpath

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--interval", default="1m")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    fetch_month(args.symbol, args.year, args.month, interval=args.interval, out_dir=args.out)

if __name__ == "__main__":
    main()
