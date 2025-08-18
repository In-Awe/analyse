import pandas as pd
from .features import add_indicators

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV and prepare it for analysis.
    Recognizes 'datetime_utc', 'timestamp', or 'open_time' for the timestamp column.
    """
    df = pd.read_csv(path)
    
    # Standardize the timestamp column name
    if 'datetime_utc' in df.columns:
        df = df.rename(columns={'datetime_utc': 'timestamp'})
    elif 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})

    if 'timestamp' not in df.columns:
        raise ValueError("CSV must contain a timestamp column named 'timestamp', 'datetime_utc', or 'open_time'")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure numeric columns are correct
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = add_indicators(df)
    return df

def df_to_context(df, lookback=60):
    recent = df.tail(lookback).copy()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_50', 'ema_200']
    available = [c for c in cols if c in recent.columns]
    return recent[available].to_dict('records')
