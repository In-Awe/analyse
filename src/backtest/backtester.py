"""Minimal vectorized backtester for 1-minute OHLCV data.
Implements:
 - Signal -> position mapping (UP/DOWN/SIDEWAYS)
 - Execution at next bar open with slippage = alpha * ATR_next
 - Binance-style fee (maker/taker) application
 - Outputs: equity_curve.csv, trades.csv, summary.json
Designed to be deterministic and readable; extend / optimize as needed.
"""
from __future__ import annotations
import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

@dataclasses.dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    fee_maker: float = 0.0002
    fee_taker: float = 0.0004
    slippage_alpha: float = 0.5   # fraction of next ATR applied as slippage
    min_notional: float = 10.0
    execution: str = "next_open"   # currently only 'next_open' supported
    pnl_col: str = "pnl"
    timestamp_col: str = "timestamp"


class Backtester:
    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        """
        df must contain:
         - timestamp (datetimeindex or column named cfg.timestamp_col)
         - open, high, low, close, volume
         - signal: integer or categorical ('UP','DOWN','SIDEWAYS') or numeric (1,-1,0)
         - atr (for slippage) recommended to be precomputed as next-bar ATR
        """
        self.cfg = cfg
        self.df = df.copy()
        # ensure timestamp index
        if cfg.timestamp_col in self.df.columns:
            self.df.index = pd.to_datetime(self.df[cfg.timestamp_col])
        else:
            if not isinstance(self.df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have timestamp column or DatetimeIndex")

        # normalize signal to -1/0/+1
        self.df["signal_num"] = self.df.get("signal", 0)
        self.df["signal_num"] = self.df["signal_num"].replace({"UP":1,"DOWN":-1,"SIDEWAYS":0})
        self.df["signal_num"] = self.df["signal_num"].astype(float).fillna(0.0)

        # placeholders for outputs
        self.trades = []  # list of dicts
        self.equity = []

    def _compute_execution_price(self, idx: pd.Timestamp) -> float:
        """
        Execution price model: execution at next bar open plus slippage = alpha * ATR_next.
        idx is the index (timestamp) of the row where we *observe* the signal; we execute on next index.
        """
        loc = self.df.index.get_loc(idx)
        if loc + 1 >= len(self.df):
            # can't execute at end of series
            return float(self.df["close"].iloc[loc])
        next_row = self.df.iloc[loc + 1]
        exec_price = float(next_row["open"])
        atr_next = float(next_row.get("atr", np.nan))
        if not math.isnan(atr_next):
            exec_price = exec_price + self.cfg.slippage_alpha * atr_next * (1 if self.df.loc[idx, "signal_num"] >= 0 else -1)
        return exec_price

    def run(self, position_sizing=None):
        """
        Vectorized-ish loop: iterate rows but avoid per-trade heavy operations.
        position_sizing: function(equity, price, signal) -> target_qty (positive for long, negative for short)
        Default sizing: fixed fraction (1% equity) converted to coin qty at execution price.
        """
        capital = self.cfg.initial_capital
        position_qty = 0.0
        position_entry_price = 0.0
        equity = capital
        equity_curve = []

        for i, ts in enumerate(self.df.index):
            row = self.df.iloc[i]
            sig = float(row["signal_num"])
            # decide target position (in units of base asset)
            if position_sizing is None:
                # default: risk 1% equity, buy as much as that buys
                risk_fraction = 0.01
                target_notional = equity * risk_fraction
            else:
                target_notional = position_sizing(equity, row, sig)

            # compute execution at next bar (if possible)
            if i + 1 >= len(self.df):
                # finalization step: mark to market
                current_price = float(row["close"])
                unreal = position_qty * (current_price - position_entry_price)
                equity = equity + unreal
                equity_curve.append({"timestamp": ts, "equity": equity})
                break

            exec_price = self._compute_execution_price(ts)

            # decide new qty
            new_qty = 0.0
            if sig > 0.5:
                new_qty = max(0.0, target_notional / max(exec_price, 1e-12))
            elif sig < -0.5:
                new_qty = -max(0.0, target_notional / max(exec_price, 1e-12))
            else:
                new_qty = 0.0

            # round to sensible precision here if needed
            # compute trade qty delta
            delta_qty = new_qty - position_qty
            trade_notional = abs(delta_qty) * exec_price
            fee = 0.0
            if trade_notional >= self.cfg.min_notional and abs(delta_qty) > 0:
                # treat all aggressive changes as taker for simplicity
                fee = trade_notional * self.cfg.fee_taker
                # apply P&L for closing portion of position
                pnl = 0.0
                if position_qty != 0:
                    # closing proportion
                    close_qty = min(abs(delta_qty), abs(position_qty)) * np.sign(position_qty)
                    pnl = -close_qty * (exec_price - position_entry_price) * -1  # careful sign
                    equity += pnl
                # update position entry price on increases (simple weighted avg)
                if abs(new_qty) > 0:
                    if position_qty == 0:
                        position_entry_price = exec_price
                    else:
                        # new weighted avg entry
                        position_entry_price = (
                            (position_entry_price * abs(position_qty) + exec_price * abs(delta_qty))
                            / max(abs(position_qty) + abs(delta_qty), 1e-12)
                        )
                position_qty = new_qty
                equity -= fee
                self.trades.append({
                    "timestamp": self.df.index[i+1] if (i+1) < len(self.df) else ts,
                    "exec_price": exec_price,
                    "delta_qty": delta_qty,
                    "position_qty": position_qty,
                    "fee": fee,
                    "equity": equity
                })

            # mark-to-market at close of that next bar
            next_close = float(self.df.iloc[i+1]["close"])
            unreal = position_qty * (next_close - position_entry_price)
            eq_snapshot = equity + unreal
            equity_curve.append({"timestamp": self.df.index[i+1], "equity": eq_snapshot})

        self.equity = pd.DataFrame(equity_curve).set_index("timestamp")
        trades_df = pd.DataFrame(self.trades)
        return self.equity, trades_df

    def save_outputs(self, outdir: str | Path):
        p = Path(outdir)
        p.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(p / "equity_curve.csv")
        pd.DataFrame(self.trades).to_csv(p / "trades.csv", index=False)
        # summary
        summary = {
            "initial_capital": self.cfg.initial_capital,
            "n_trades": len(self.trades),
            "final_equity": float(self.equity["equity"].iloc[-1]) if len(self.equity) else float(self.cfg.initial_capital)
        }
        with open(p / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # quick local smoke runner if executed directly (for development)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="cleaned BTCUSD_1min.csv")
    parser.add_argument("--out", default="artifacts/backtest/default", help="output dir")
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    # Ensure atr exists; simple fallback
    if "atr" not in df.columns:
        df["returns"] = np.log(df["close"]).diff().fillna(0)
        df["atr"] = df["returns"].rolling(14).std().fillna(0) * df["close"]

    cfg = BacktestConfig(initial_capital=args.capital)
    bt = Backtester(df, cfg)
    equity, trades = bt.run()
    bt.save_outputs(args.out)
