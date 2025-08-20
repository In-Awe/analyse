import os
from backend.data.loader import load_csv

def test_load_sample(tmp_path):
    path = tmp_path / "sample_minute.csv"
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='T'),
        'open': (100 + np.arange(10)).astype(float),
        'high': (101 + np.arange(10)).astype(float),
        'low': (99 + np.arange(10)).astype(float),
        'close': (100.5 + np.arange(10)).astype(float),
        'volume': (1.0 + np.random.rand(10))
    })
    df.to_csv(path, index=False)
    df_loaded = load_csv(str(path))
    assert 'rsi' in df_loaded.columns
    assert len(df_loaded) == 10
