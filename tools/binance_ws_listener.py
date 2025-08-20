#!/usr/bin/env python3
"""
tools/binance_ws_listener.py

Simple listener to receive closed 1m kline events and append them to CSV (or DB ingestion).
Requires: python-binance, pandas

Usage:
  pip install python-binance pandas
  export BINANCE_API_KEY=...
  export BINANCE_API_SECRET=...
  python tools/binance_ws_listener.py
"""
import os
import time
import pandas as pd
from binance import ThreadedWebsocketManager

OUT_DIR = os.getenv("OUT_DIR", "data/raw")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").upper()
OUT_CSV = os.path.join(OUT_DIR, f"{SYMBOL}_live_1m.csv")

def on_kline(msg):
    try:
        if msg.get('e') != 'kline':
            return
        k = msg['k']
        # if candle closed
        if k.get('x'):
            row = {
                'open_time': pd.to_datetime(k['t'], unit='ms', utc=True),
                'open': k['o'],
                'high': k['h'],
                'low': k['l'],
                'close': k['c'],
                'volume': k['v'],
                'close_time': pd.to_datetime(k['T'], unit='ms', utc=True),
                'num_trades': k.get('n', None),
            }
            df = pd.DataFrame([row])
            os.makedirs(OUT_DIR, exist_ok=True)
            if not os.path.exists(OUT_CSV):
                df.to_csv(OUT_CSV, index=False)
            else:
                df.to_csv(OUT_CSV, index=False, header=False, mode='a')
            print("Appended closed candle:", row['open_time'])
    except Exception as e:
        print("Error in on_kline:", e)

def main():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()
    print("Starting kline socket for", SYMBOL)
    twm.start_kline_socket(callback=on_kline, symbol=SYMBOL, interval='1m')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping websocket manager")
        twm.stop()

if __name__ == "__main__":
    main()
