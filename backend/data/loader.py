import pandas as pd
from .features import add_indicators

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns and 'open_time' in df.columns:
        df = df.rename(columns={'open_time':'timestamp'})
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        else:
            raise ValueError("CSV missing timestamp/open_time")
    df = df.sort_values('timestamp').reset_index(drop=True)
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = add_indicators(df)
    return df

def df_to_context(df, lookback=60):
    recent = df.tail(lookback).copy()
    cols = ['timestamp','open','high','low','close','volume','rsi','ema_50','ema_200']
    available = [c for c in cols if c in recent.columns]
    return recent[available].to_dict('records')
