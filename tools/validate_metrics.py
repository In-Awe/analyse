#!/usr/bin/env python3
"""
tools/validate_metrics.py

Recompute performance metrics (Sharpe, Sortino, MaxDrawdown, Profit Factor, etc.)
from equity_curve.csv and trades.csv. Prints a JSON object of metrics.

Usage:
  python tools/validate_metrics.py <equity_csv> [trades_csv]

Notes:
 - Expects the equity CSV to have a numeric equity column (named 'equity' or second column).
 - Expects trades CSV to have a 'pnl' column. If absent, PF may be NaN.
"""
import sys
import pandas as pd
import numpy as np
import json

MINUTES_PER_YEAR = 525600  # 365*24*60

def sharpe(returns):
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float('nan')
    return (r.mean() * np.sqrt(MINUTES_PER_YEAR)) / r.std()

def sortino(returns):
    r = returns.dropna()
    neg = r[r < 0]
    if len(r) < 2 or neg.std() == 0:
        return float('nan')
    downside = neg.std()
    return (r.mean() * np.sqrt(MINUTES_PER_YEAR)) / downside

def max_drawdown(equity):
    eq = equity.fillna(method='ffill')
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    mdd = drawdown.min()
    return float(mdd)

def profit_factor(trades_pnl):
    wins = trades_pnl[trades_pnl > 0].sum()
    losses = -trades_pnl[trades_pnl < 0].sum()
    if losses == 0:
        return float('inf') if wins > 0 else float('nan')
    return float(wins / losses)

def compute_metrics_from_equity(equity_csv, trades_csv=None):
    df = pd.read_csv(equity_csv)
    # detect equity column
    if 'equity' in df.columns:
        equity = df['equity']
    elif 'value' in df.columns:
        equity = df['value']
    elif df.shape[1] >= 2:
        equity = df.iloc[:,1]
    else:
        raise ValueError("Could not detect equity column in " + str(equity_csv))
    pct = equity.pct_change().replace([np.inf, -np.inf], np.nan)
    metrics = {
        'final_equity': float(equity.iloc[-1]) if not equity.empty else float('nan'),
        'sharpe': float(sharpe(pct)),
        'sortino': float(sortino(pct)),
        'max_drawdown': float(max_drawdown(equity)),
    }
    if trades_csv:
        try:
            trades = pd.read_csv(trades_csv)
            if 'pnl' in trades.columns:
                pnl = trades['pnl']
            elif 'profit' in trades.columns:
                pnl = trades['profit']
            else:
                pnl = pd.Series(dtype=float)
            metrics.update({
                'profit_factor': profit_factor(pnl) if len(pnl)>0 else float('nan'),
                'win_rate': float((pnl > 0).sum() / len(pnl)) if len(pnl)>0 else float('nan'),
                'avg_win': float(pnl[pnl>0].mean()) if (pnl>0).any() else float('nan'),
                'avg_loss': float(pnl[pnl<0].mean()) if (pnl<0).any() else float('nan'),
            })
        except FileNotFoundError:
            # This is not an error, trades_csv is optional
            pass
        except Exception as e:
            print(f"Could not process trades_csv {trades_csv}: {e}", file=sys.stderr)

    return metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_metrics.py <equity_csv> [trades_csv]", file=sys.stderr)
        sys.exit(1)
    equity_csv = sys.argv[1]
    trades_csv = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        metrics = compute_metrics_from_equity(equity_csv, trades_csv)
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
