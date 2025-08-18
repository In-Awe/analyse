import pandas as pd
import numpy as np
from phase1_data_audit_pipeline import add_basic_features, advanced_spike_detection

def make_synthetic_series(n_minutes=240, base_price=30000.0, seed=42):
    np.random.seed(seed)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_minutes, freq='min')
    rets = np.random.normal(loc=0.0, scale=0.0002, size=n_minutes)
    prices = base_price * np.exp(np.cumsum(rets))
    df = pd.DataFrame(index=idx)
    df['open'] = prices
    df['high'] = prices * (1 + np.abs(np.random.normal(0, 0.0005, size=n_minutes)))
    df['low'] = prices * (1 - np.abs(np.random.normal(0, 0.0005, size=n_minutes)))
    df['close'] = prices
    df['volume'] = np.random.randint(1, 10, size=n_minutes)
    return df

def inject_spike(df, minute_index=120, price_jump=0.02, vol_multiplier=30):
    df = df.copy()
    df.iloc[minute_index, df.columns.get_loc('close')] *= (1 + price_jump)
    df.iloc[minute_index, df.columns.get_loc('high')] = df.iloc[minute_index]['close'] * 1.0005
    df.iloc[minute_index, df.columns.get_loc('low')] = df.iloc[minute_index]['close'] * 0.9995
    df.iloc[minute_index, df.columns.get_loc('open')] = df.iloc[minute_index]['close'] / (1 + price_jump)
    df.iloc[minute_index, df.columns.get_loc('volume')] *= vol_multiplier
    return df

def test_detect_injected_spike():
    df = make_synthetic_series(n_minutes=240)
    df = inject_spike(df, minute_index=120, price_jump=0.02, vol_multiplier=50)
    df = add_basic_features(df)
    spikes = advanced_spike_detection(df, window=60, min_volume=1, ret_mad_thresh=4.0, vol_surge_thresh=3.0)
    assert not spikes.empty, "Spike detector failed to find the injected spike"
    injected_ts = df.index[120]
    # Convert spikes.index to a list for the assertion
    assert injected_ts in spikes.index.tolist(), "Injected spike timestamp not found"
