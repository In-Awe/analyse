#!/usr/bin/env python3
"""
tools/binance_public_fetcher.py

Public Binance klines fetcher. Uses REST endpoints without requiring API keys.
Provides functions:
  - get_all_symbols()
  - fetch_month_klines(symbol, year, month, interval='1m')
  - download_symbol_history(symbol, out_dir, months_back=84, interval='1m')

Notes:
 - This script fetches month-by-month to avoid over-requesting and to allow
   detecting when a coin didn't exist (empty months).
 - Output CSV format: use the same column layout used elsewhere in repo:
   open_time,open,high,low,close,volume,close_time,quote_asset_volume,num_trades,taker_buy_base,taker_buy_quote,ignore
"""
from __future__ import annotations
import requests
import time
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional

BASE = "https://api.binance.com"

def get_all_symbols() -> List[str]:
    """Return a list of symbol strings (e.g., 'BTCUSDT') from exchangeInfo."""
    url = f"{BASE}/api/v3/exchangeInfo"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    symbols = [s['symbol'] for s in j.get('symbols', [])]
    return symbols

def month_range_iter(months_back: int = 84):
    """Yield (year, month, start_ts_ms, end_ts_ms) tuples for the last months_back months."""
    now = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    for i in range(months_back):
        start = (now - timedelta(days=30*i)).replace(day=1)
        # compute end: next month first day
        if start.month == 12:
            end = start.replace(year=start.year+1, month=1)
        else:
            end = start.replace(month=start.month+1)
        # yield from oldest to newest
        yield start.year, start.month, int(start.timestamp()*1000), int(end.timestamp()*1000)

def fetch_month_klines(symbol: str, start_time_ms: int, end_time_ms: int, interval='1m', limit=1000, api_key: Optional[str] = None):
    """Fetch klines for symbol between start and end times (ms). Returns list of klines."""
    url = f"{BASE}/api/v3/klines"
    headers = {}
    if api_key:
        headers['X-MBX-APIKEY'] = api_key
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time_ms,
        'endTime': end_time_ms,
        'limit': limit
    }
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data

def klines_to_df(klines):
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(klines, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    return df

def download_symbol_history(symbol: str, out_dir: str = "artifacts/raw", months_back: int = 84, interval='1m', api_key: Optional[str]=None, sleep_sec: float=0.35):
    """
    Download month-by-month history for a symbol and write per-month CSVs to:
      {out_dir}/{symbol}/{symbol}_1m_{YYYY-MM}.csv
    Stop early if a month returns 0 rows (indicating the coin may not have existed).
    """
    symbol_dir = Path(out_dir) / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    months = list(month_range_iter(months_back))
    months.reverse()  # oldest first
    got_any = False
    for (year, month, st_ms, en_ms) in months:
        outfn = symbol_dir / f"{symbol}_1m_{year:04d}-{month:02d}.csv"
        if outfn.exists() and outfn.stat().st_size > 100:
            # skip if already present and non-empty
            print(f"Skipping existing {outfn}")
            got_any = True
            continue
        try:
            klines = fetch_month_klines(symbol, st_ms, en_ms, interval=interval, api_key=api_key)
        except requests.HTTPError as e:
            print(f"HTTP error fetching {symbol} {year}-{month}: {e}")
            time.sleep(sleep_sec*2)
            continue
        except Exception as e:
            print(f"Error fetching {symbol} {year}-{month}: {e}")
            time.sleep(sleep_sec*2)
            continue
        if not klines:
            # no data for this month
            print(f"No klines for {symbol} {year}-{month}; stopping historic fetch for older months.")
            # If we've already retrieved newer months, continue to next (since we iterate oldest->newest),
            # but stop if this was the first month (coin likely not yet existed)
            if not got_any:
                print(f"No historical data found at all for {symbol}; skipping.")
            break
        df = klines_to_df(klines)
        df.to_csv(outfn, index=False)
        got_any = True
        print(f"Wrote {outfn} rows={len(df)}")
        time.sleep(sleep_sec)  # respect rate limits
    return got_any
