"""Simple replay script to emit candle events from a cleaned CSV for local integration tests.
"""
import time
import argparse
from typing import Callable
import pandas as pd

def emit_row(row: pd.Series, emitter: Callable[[dict], None]):
    # convert row fields to types
    candle = {
       'type': 'candle',
       'symbol': row.get('symbol','BTCUSDT'),
       'ts': int(row.name.timestamp() * 1000), # Convert timestamp to milliseconds
       'open': float(row['open']),
       'high': float(row['high']),
       'low': float(row['low']),
       'close': float(row['close']),
       'volume': float(row.get('volume',0))
    }
    emitter(candle)

def replay(csv_path: str, emitter: Callable[[dict], None], realtime: bool = False, rate: float = 1.0):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    for _, row in df.iterrows():
       emit_row(row, emitter)
       if realtime:
           time.sleep(60.0 / rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Path to cleaned candles CSV')
    parser.add_argument('--realtime', action='store_true')
    args = parser.parse_args()
    def print_ev(e):
       print(e)
    replay(args.csv, print_ev, realtime=args.realtime)
