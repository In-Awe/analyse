import os
import pandas as pd
from binance.client import Client

class BinanceClient:
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        api_key = api_key or os.getenv("BINANCE_API_KEY")
        api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        if testnet:
            api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET")
        # Client will accept None for public endpoints, but for private calls keys are required
        self.client = Client(api_key, api_secret)

    def get_historical_klines(self, symbol, start_str, end_str, interval='1m') -> pd.DataFrame:
        """
        Returns a DataFrame with columns timestamp, open, high, low, close, volume
        start_str and end_str are strings like '2024-11-01' or '1 day ago UTC'
        """
        raw = self.client.get_historical_klines(symbol, interval, start_str, end_str)
        cols = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore']
        df = pd.DataFrame(raw, columns=cols)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df[['timestamp','open','high','low','close','volume']]

    # Future: implement testnet create_test_order and live order wrappers as needed.
