# Phase IV — Backtesting & Robustness

This document explains the basic backtesting runner added in `src/backtest` and how to run the smoke tests.

## Files added
- `src/backtest/backtester.py` — minimal vectorized engine, execution at next bar open + slippage, fee model.
- `src/backtest/metrics.py` — Sharpe, Sortino, MDD, Profit Factor, Win Rate helpers.
- `src/backtest/robustness.py` — Monte Carlo reshuffle and noise-addition tests.
- `scripts/run_backtest.py` — CLI to run the backtest using `configs/backtest.yaml`.
- `configs/backtest.yaml` — default backtest + robustness parameters.

## Run a smoke backtest
1. Ensure `data/cleaned/BTCUSD_1min.cleaned.csv` exists (Phase I produces this).
2. From repo root:
```bash
python -m scripts.run_backtest --data data/cleaned/BTCUSD_1min.cleaned.csv --out artifacts/backtest/run1
```
3. Outputs:
 - `artifacts/backtest/run1/equity_curve.csv`
 - `artifacts/backtest/run1/trades.csv`
 - `artifacts/backtest/run1/summary.json`

## Robustness
Use `src/backtest/robustness.py` to:
 - produce Monte Carlo reshuffle statistics over per-trade P&L (requires trades.csv with 'pnl' column).
 - run noise trials against the price series and produce stability metrics.

## Next steps (suggested)
1. Hook unit tests around the replay smoke to ensure end-to-end path (not included in patch).
2. Fill the `signal` column using the Phase III model artifact and run OOS/walk-forward harness.
3. Add trade-level pnl calculation to `backtester` (currently skeletoned as list entries) and expose more realistic slippage/limit simulation when tick-level data is available.
