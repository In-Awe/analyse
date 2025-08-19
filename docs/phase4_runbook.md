# Phase IV Runbook

## TL;DR
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_backtest.py --config configs/trade_logic.yaml
python scripts/run_robustness.py --all
python scripts/plot_equity.py --equity-csv artifacts/backtest/equity_curve.csv
```

## Inputs
- Cleaned minute data: `data/cleaned/BTCUSD_1min.cleaned.csv` (Phase I)
- Features parquet: `data/features/microstructure.parquet` (Phase II)
- Final model artifact (optional; falls back to heuristic): `artifacts/models/final_model_xgb.pkl` (Phase III)

## Outputs
- `artifacts/backtest/equity_curve.csv|png`, `trades.csv`, `summary.json`
- `artifacts/robustness/*` with:
  - `oos_summary.json`
  - `walk_forward_summary.json`
  - `monte_carlo_summary.json`
  - `sensitivity.csv`, `sensitivity_summary.json`
  - `noise_summary.json`
  - `gate_summary.json`

## Fees & Slippage
- Fees: configurable maker/taker in **bps** via `configs/trade_logic.yaml`.
- Slippage: ATR-based with `alpha` multiplier or simple next-open fill.

## Acceptance Gate (Phase IV)
Mark “candidate for Phase V” iff **all**:
1. `summary.sharpe >= 1.0`
2. `summary.profit_factor >= 1.2`
3. Monte Carlo `median_total_pnl` **> 0**
4. `sensitivity_summary.cliff_like_sensitivity == false`

If any fails → **Requires Further Research**. See `artifacts/robustness/gate_summary.json`.

## Notes
- Walk-forward can evaluate with fixed model (default) or `retrain: true` to simulate rolling re‑fit (hook present).
- If ATR column missing in cleaned data, a 14‑EMA ATR is computed automatically.
