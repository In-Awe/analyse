from __future__ import annotations
import pandas as pd
from src.backtest.backtester import VectorizedBacktester

def test_min_notional_enforced(tmp_path):
    cfg = {
        "fees": {"side": "taker", "taker_bps": 10.0, "maker_bps": 2.0},
        "slippage": {"model": "next_open", "alpha": 0.0},
        "execution": {"fill_on": "next_open", "tick_size": 0.01, "lot_size": 0.001},
        "positioning": {"min_notional_usd": 100.0, "up": 1.0, "down": -1.0, "sideways": 0.0},
        "paths": {"outputs_dir": str(tmp_path)}
    }
    b = VectorizedBacktester(cfg, outputs_dir=str(tmp_path))
    idx = pd.date_range("2024-01-01", periods=5, freq="T")
    df = pd.DataFrame({
        "open":[10,10,10,10,10],
        "high":[10,10,10,10,10],
        "low" :[10,10,10,10,10],
        "close":[10,10,10,10,10],
        "volume":[1,1,1,1,1],
        "atr":[0.1]*5
    }, index=idx)
    sig = pd.Series(["SIDEWAYS","UP","UP","SIDEWAYS","SIDEWAYS"], index=idx)
    res = b.run(df, sig)
    trades = res["trades"]
    assert (trades["qty"] * trades["exec_price"]).min() >= 100.0
