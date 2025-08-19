# Backtesting & Phase IV — Vectorized Backtester (phenol Phase 4 scaffold)

This module provides a vectorized backtester skeleton and a smoke CI workflow.

Files added:
- `src/backtest/backtester.py` — vectorized backtester (pandas).
- `scripts/run_backtest.py` — CLI runner for local/backtest runs.
- `configs/trade_logic.yaml` — default config for fees/slippage/execution.
- `.github/workflows/backtest_smoke.yml` — smoke GitHub Action.

Outputs:
- `artifacts/backtest/equity_curve.csv`
- `artifacts/backtest/trades.csv`
- `artifacts/backtest/summary.json`

Execution (local):
```bash
pip install -r requirements-backtest.txt
python scripts/run_backtest.py --input /data/cleaned/BTCUSD_1min.cleaned.csv --outdir artifacts/backtest/run_YYYYMMDD_HHMM
```

Notes:

The backtester assumes minute OHLCV and uses the next-bar open as base execution price,
with configurable slippage = alpha * ATR_next_bar.

Fees are applied per trade (maker/taker). Execution mode and min_notional are configurable.

This is a skeleton suitable for extension: Monte Carlo reshuffles, walk-forward loops,
parameter sweeps, and more robust slippage/limit-order filling models should be added next.


---

### `requirements-backtest.txt`


pandas>=2.0
numpy>=1.25
pyyaml
scipy
statsmodels
