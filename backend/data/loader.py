import pandas as pd
from .features import add_indicators

def load_csv(path: str) -> pd.DataFrame:
    """
    Expect CSV with at least: timestamp, open, high, low, close, volume
    Timestamps will be parsed to pandas datetime
    """
    df = pd.read_csv(path)
    # support common column variants
    if 'timestamp' not in df.columns and 'open_time' in df.columns:
        df = df.rename(columns={'open_time':'timestamp'})
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        # attempt epoch ms fallback
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        else:
            raise ValueError("CSV missing timestamp or open_time")
    df = df.sort_values('timestamp').reset_index(drop=True)
    # coerce numeric columns
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = add_indicators(df)
    return df

def df_to_context(df: pd.DataFrame, lookback: int = 60) -> list:
    """
    Return last `lookback` rows as list of dicts for the prompt builder
    """
    recent = df.tail(lookback).copy()
    cols = ['timestamp','open','high','low','close','volume','rsi','ema_50','ema_200']
    available = [c for c in cols if c in recent.columns]
    records = recent[available].to_dict('records')
    return records
