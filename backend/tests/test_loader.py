from backend.data.loader import load_csv
def test_load_sample(tmp_path):
    p = tmp_path / "s.csv"
    import pandas as pd
    df = pd.DataFrame({
      'timestamp': pd.date_range('2025-01-01', periods=5, freq='T'),
      'open': [1,2,3,4,5],
      'high': [1,2,3,4,5],
      'low': [1,2,3,4,5],
      'close': [1,2,3,4,5],
      'volume':[1,1,1,1,1]
    })
    df.to_csv(p, index=False)
    df2 = load_csv(str(p))
    assert 'rsi' in df2.columns
