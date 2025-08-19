from __future__ import annotations
import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 365 * 24 * 60

def _to_series(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)

def equity_from_pnl(pnl_series: pd.Series, initial_equity: float = 10000.0) -> pd.Series:
    eq = pnl_series.cumsum() + initial_equity
    return eq

def drawdown(equity: pd.Series) -> pd.Series:
    peaks = equity.cummax()
    dd = (equity - peaks) / peaks.replace(0, np.nan)
    return dd.fillna(0.0)

def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())

def sharpe(returns_minute: pd.Series, risk_free: float = 0.0) -> float:
    # minute returns -> annualized
    r = _to_series(returns_minute).dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return 0.0
    mean = r.mean() - risk_free / MINUTES_PER_YEAR
    return float(mean / r.std(ddof=1) * np.sqrt(MINUTES_PER_YEAR))

def sortino(returns_minute: pd.Series, risk_free: float = 0.0) -> float:
    r = _to_series(returns_minute).dropna()
    downside = r[r < 0]
    if downside.std(ddof=1) == 0 or len(r) < 2:
        return 0.0
    mean = r.mean() - risk_free / MINUTES_PER_YEAR
    return float(mean / downside.std(ddof=1) * np.sqrt(MINUTES_PER_YEAR))

def calmar(equity: pd.Series) -> float:
    eq = _to_series(equity)
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0 if len(eq) > 1 else 0.0
    mdd = abs(max_drawdown(eq))
    if mdd == 0:
        return 0.0
    # approximate per-year return from total; acceptable for Phase IV
    years = max(1.0, (len(eq) / MINUTES_PER_YEAR))
    ann_return = (1 + total_return) ** (1/years) - 1.0
    return float(ann_return / mdd)

def profit_factor(trade_pnl: pd.Series) -> float:
    gains = trade_pnl[trade_pnl > 0].sum()
    losses = -trade_pnl[trade_pnl < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)

def win_rate(trade_pnl: pd.Series) -> float:
    n = len(trade_pnl)
    if n == 0:
        return 0.0
    return float((trade_pnl > 0).sum() / n)

def avg_win_loss(trade_pnl: pd.Series) -> tuple[float, float]:
    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    return float(wins.mean() if len(wins) else 0.0), float(losses.mean() if len(losses) else 0.0)

def summarize(equity: pd.Series, minute_returns: pd.Series, trade_pnl: pd.Series) -> dict:
    return {
        "sharpe": sharpe(minute_returns),
        "sortino": sortino(minute_returns),
        "calmar": calmar(equity),
        "max_drawdown": float(max_drawdown(equity)),
        "profit_factor": profit_factor(trade_pnl),
        "win_rate": win_rate(trade_pnl),
        "avg_win": avg_win_loss(trade_pnl)[0],
        "avg_loss": avg_win_loss(trade_pnl)[1],
        "total_trades": int(len(trade_pnl)),
        "final_equity": float(equity.iloc[-1]),
    }
