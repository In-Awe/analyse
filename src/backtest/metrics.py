#!/usr/bin/env python3
"""
Backtest metrics helper functions.
Provides Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, Win Rate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def annualize_return(returns: np.ndarray, period_per_year: int) -> float:
    # simple compounded annual growth rate approximation
    cum = np.prod(1 + returns) - 1
    years = len(returns) / period_per_year
    if years <= 0:
        return 0.0
    try:
        return (1 + cum) ** (1 / years) - 1
    except Exception:
        return 0.0

def sharpe_ratio(returns: np.ndarray, period_per_year: int = 365*24*60, risk_free_rate: float = 0.0) -> float:
    # returns: array of periodic returns (fractional)
    mean_r = np.mean(returns) - (risk_free_rate / period_per_year)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0
    return (mean_r * np.sqrt(period_per_year)) / std_r

def sortino_ratio(returns: np.ndarray, period_per_year: int = 365*24*60, required_return: float = 0.0) -> float:
    downside = returns[returns < required_return / period_per_year]
    if len(downside) == 0:
        return 0.0
    expected_return = np.mean(returns) - (required_return / period_per_year)
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std == 0:
        return 0.0
    return (expected_return * np.sqrt(period_per_year)) / downside_std

def max_drawdown(equity_curve: pd.Series) -> float:
    # equity_curve: series of portfolio values by time index
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min() if len(drawdown) > 0 else 0.0

def calmar_ratio(returns: np.ndarray, equity_curve: pd.Series, period_per_year: int = 365*24*60) -> float:
    ann_return = annualize_return(returns, period_per_year)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return ann_return / mdd

def profit_factor(trades: pd.DataFrame) -> float:
    # trades df expected to have 'pnl' column (monetary)
    wins = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    losses = -trades.loc[trades['pnl'] < 0, 'pnl'].sum()
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins) / float(losses)

def win_rate(trades: pd.DataFrame) -> float:
    if len(trades) == 0:
        return 0.0
    return float((trades['pnl'] > 0).sum()) / len(trades)

def avg_win_loss(trades: pd.DataFrame) -> Dict[str, float]:
    wins = trades.loc[trades['pnl'] > 0, 'pnl']
    losses = trades.loc[trades['pnl'] < 0, 'pnl']
    return {
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
    }

def summary_from_returns(equity: pd.Series, returns: np.ndarray, trades: pd.DataFrame, period_per_year: int = 365*24*60) -> Dict:
    s = {}
    s["sharpe"] = float(sharpe_ratio(returns, period_per_year))
    s["sortino"] = float(sortino_ratio(returns, period_per_year))
    s["max_drawdown"] = float(max_drawdown(equity))
    s["calmar"] = float(calmar_ratio(returns, equity, period_per_year))
    s["profit_factor"] = float(profit_factor(trades))
    s["win_rate"] = float(win_rate(trades))
    s.update(avg_win_loss(trades))
    s["total_trades"] = int(len(trades))
    s["ending_equity"] = float(equity.iloc[-1]) if len(equity) > 0 else 0.0
    return s
