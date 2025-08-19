import numpy as np
import pandas as pd

def sharpe_ratio(returns, annualization=365*24*60):
    """returns: pd.Series of minute returns (fractional). annualization default: minutes/year"""
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0:
        return float("nan")
    return (mean * annualization) / (std * np.sqrt(annualization))

def sortino_ratio(returns, required_return=0.0, annualization=365*24*60):
    downside = returns[returns < required_return]
    if len(downside) == 0:
        return float("nan")
    dr = np.sqrt((downside ** 2).mean()) * np.sqrt(annualization)
    if dr == 0:
        return float("nan")
    rp = (returns.mean() - required_return) * annualization
    return rp / dr

def max_drawdown(equity_series: pd.Series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    mdd = drawdown.min()
    return float(abs(mdd))

def profit_factor(trade_pnl):
    wins = trade_pnl[trade_pnl > 0].sum()
    losses = -trade_pnl[trade_pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)

def win_rate(trade_pnl):
    total = len(trade_pnl)
    if total == 0:
        return float("nan")
    return float((trade_pnl > 0).sum() / total)
