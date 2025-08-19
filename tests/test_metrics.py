from __future__ import annotations
import pandas as pd
from src.backtest.metrics import sharpe, sortino, profit_factor, equity_from_pnl

def test_metrics_shapes():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert isinstance(sharpe(r), float)
    assert isinstance(sortino(r), float)
    pnl = pd.Series([1, -0.5, 2, 0])
    assert profit_factor(pnl) > 0
    eq = equity_from_pnl(pnl)
    assert len(eq) == len(pnl)
