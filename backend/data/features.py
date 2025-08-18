import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if 'close' not in df.columns:
        return df
    try:
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_200'] = ta.ema(df['close'], length=200)
    except Exception:
        df['rsi'] = pd.NA
        df['ema_50'] = pd.NA
        df['ema_200'] = pd.NA
    return df
