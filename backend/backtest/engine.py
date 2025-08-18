import pandas as pd
from .metrics import compute_metrics

def simple_ma_rsi_strategy(df: pd.DataFrame):
    """
    Example strategy:
    - Buy when EMA50 crosses above EMA200 and RSI < 70
    - Sell when EMA50 crosses below EMA200 or RSI > 70
    This is a naive backtest for demonstration.
    """
    df = df.copy().reset_index(drop=True)
    df['position'] = 0
    for i in range(1, len(df)):
        prev = df.loc[i-1]
        cur = df.loc[i]
        buy_signal = (prev.get('ema_50') <= prev.get('ema_200')) and (cur.get('ema_50') > cur.get('ema_200')) and (cur.get('rsi') is not None and cur.get('rsi') < 70)
        sell_signal = ((prev.get('ema_50') >= prev.get('ema_200')) and (cur.get('ema_50') < cur.get('ema_200'))) or (cur.get('rsi') is not None and cur.get('rsi') > 70)
        if buy_signal:
            df.at[i, 'position'] = 1
        elif sell_signal:
            df.at[i, 'position'] = 0
        else:
            df.at[i, 'position'] = df.at[i-1, 'position']

    # compute returns based on close price change
    df['close_shift'] = df['close'].shift(1)
    df['strategy_ret'] = 0.0
    mask = df['close_shift'] != 0
    df.loc[mask, 'strategy_ret'] = ((df.loc[mask, 'close'] - df.loc[mask, 'close_shift']) / df.loc[mask, 'close_shift']) * df.loc[mask, 'position']
    return compute_metrics(df['strategy_ret'])

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    # quick demo
    df = pd.DataFrame({
        'close': 100 + (np.arange(200) * 0.1),
        'ema_50': 100 + (np.arange(200) * 0.09),
        'ema_200': 100 + (np.arange(200) * 0.05),
        'rsi': [50]*200
    })
    print(simple_ma_rsi_strategy(df))
