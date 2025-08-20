from __future__ import annotations
import math
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .metrics import equity_from_pnl, summarize


class VectorizedBacktester:
    """
    Vectorized minute-level backtester with configurable fees/slippage and
    signal-to-position mapping. Avoids explicit trade loops where possible.
    """
    def __init__(self, cfg: Dict, outputs_dir: str = "artifacts/backtest/"):
        self.cfg = cfg
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _bps_to_frac(bps: float) -> float:
        return bps / 1e4

    def _effective_fee_frac(self) -> float:
        side = self.cfg["fees"]["side"]
        if side == "maker":
            return self._bps_to_frac(self.cfg["fees"]["maker_bps"])
        return self._bps_to_frac(self.cfg["fees"]["taker_bps"])

    def _apply_slippage(self, df: DataFrame) -> Series:
        mode = self.cfg["slippage"]["model"]
        if self.cfg["execution"]["fill_on"] == "next_open":
            price = df["open"].shift(-1)
        else:
            price = df["close"]

        if mode == "next_open":
            return price
        # ATR-based slip: price ± alpha * ATR_next
        alpha = float(self.cfg["slippage"]["alpha"])
        if "atr" not in df.columns:
            # safe fallback
            return price
        atr_next = df["atr"].shift(-1)
        # sign of trade decided by position change; handled later
        return price, atr_next, alpha

    def _normalize_qty(self, qty: Series, lot_size: float) -> Series:
        # round down to lot size
        return (np.floor(qty / lot_size) * lot_size).astype(float)

    def run(self, df: DataFrame, signals: Series, initial_equity: float = 10000.0) -> Dict:
        """
        df: minute OHLCV with at least columns ['open','high','low','close','volume'] and 'atr' if slippage=atr
        signals: predicted class label per minute in {"UP","DOWN","SIDEWAYS"}
        """
        cfg = self.cfg
        df = df.copy()
        df["signal"] = signals.reindex(df.index).fillna("SIDEWAYS")

        # Map signal -> target position
        pos_map = {"UP": cfg["positioning"]["up"],
                   "DOWN": cfg["positioning"]["down"],
                   "SIDEWAYS": cfg["positioning"]["sideways"]}
        df["target_pos"] = df["signal"].map(pos_map).fillna(0.0)

        # Position changes -> trades at next bar open (or configured)
        df["pos_prev"] = df["target_pos"].shift(1).fillna(0.0)
        df["pos_change"] = df["target_pos"] - df["pos_prev"]

        # Notional sizing: 1 unit == 1 BTC equivalent; enforce min notional
        tick_size = float(cfg["execution"]["tick_size"])
        lot_size = float(cfg["execution"]["lot_size"])
        min_notional = float(cfg["positioning"]["min_notional_usd"])

        # Fill price & potential ATR slip
        slip = self._apply_slippage(df)
        if isinstance(slip, tuple):
            base_price, atr_next, alpha = slip
        else:
            base_price = slip
            atr_next, alpha = (pd.Series(0.0, index=df.index), 0.0)

        # trade_sign: +1 for buy (increase pos), -1 for sell (decrease pos)
        trade_sign = np.sign(df["pos_change"]).astype(float)

        # execution price adjusted by slippage in direction of trade
        exec_price = base_price + trade_sign * alpha * atr_next
        exec_price = exec_price.round(int(max(0, -math.log10(tick_size))))  # quantize to tick

        # Quantity in "units" (e.g., BTC). Here, pos is already expressed in units.
        qty = df["pos_change"].abs()
        # Enforce min notional where there is a non-zero change
        notional = qty * exec_price
        small = (notional > 0) & (notional < min_notional)
        qty = qty.where(~small, min_notional / exec_price)
        qty = self._normalize_qty(qty, lot_size)

        fee_frac = self._effective_fee_frac()
        fees = qty * exec_price * fee_frac

        # PnL is decomposed into holding PnL and trading PnL.
        # Holding PnL: PnL from the position held at the start of the bar, marked to market.
        # Trading PnL: PnL from the change in position during the bar.
        pnl_from_holding = df["pos_prev"] * (df["close"] - df["close"].shift(1))
        pnl_from_trading = df["pos_change"] * (df["close"] - exec_price)
        minute_pnl = pnl_from_holding.fillna(0) + pnl_from_trading.fillna(0) - fees.fillna(0)

        equity = equity_from_pnl(minute_pnl, initial_equity=initial_equity)

        # Aggregate trade-level pnl for summary: consider only non-zero pos_change rows
        trades = df.loc[qty > 0, ["signal", "pos_prev", "target_pos"]].copy()
        trades["qty"] = qty[qty > 0]
        trades["exec_price"] = exec_price[qty > 0]
        trades["fees"] = fees[qty > 0]
        trades["direction"] = trade_sign[qty > 0]

        # Trade PnL is the sum of minute-level PnLs over the lifetime of the trade.
        # This requires a trade-closing logic which is complex in vectorized form.
        # As a better approximation, we attribute the minute_pnl of the execution bar to the trade.
        # Note: sum(trades.pnl) will not equal (final_equity - initial_equity) because of the
        # PnL from holding positions on bars without trades. This is expected in this simplified model.
        trades["pnl"] = minute_pnl[qty > 0]

        # Metrics
        # Prepend initial equity for correct return calculation
        equity_with_initial = pd.concat([pd.Series([initial_equity], index=[df.index[0] - pd.Timedelta(minutes=1)]), equity])
        minute_returns = minute_pnl / equity.shift(1).fillna(initial_equity)
        summary = summarize(equity_with_initial, minute_returns, trades["pnl"])

        # Persist
        outdir = Path(self.cfg["paths"]["outputs_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({
            "equity": equity,
            "minute_pnl": minute_pnl,
            "minute_return": minute_returns,
            "pos": df["target_pos"],
            "pos_change": df["pos_change"],
            "exec_price": exec_price,
        }, index=df.index)
        # Prepend initial equity to the CSV for correct downstream calculations
        df_out.loc[df.index[0] - pd.Timedelta(minutes=1)] = [initial_equity, 0, 0, 0, 0, 0]
        df_out = df_out.sort_index()

        _save_backtest_outputs_safely(df_out, trades, out_dir=outdir)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        return {
            "equity_curve": df_out,
            "trades": trades,
            "summary": summary
        }


def load_cfg(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_cleaned(csv_path: str) -> DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = df.sort_index()
    # sanity columns
    needed = ["open","high","low","close","volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "atr" not in df.columns:
        # compute simple ATR if absent (Phase II should have provided)
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df

def load_signals(cfg: Dict, df: DataFrame) -> Series:
    """
    Phase IV runs on the selected model artifact from Phase III.
    If unavailable, fall back to a transparent heuristic: UP when 10-30 EMA diff positive and ATR regime low.
    """
    model_path = Path(cfg["paths"]["model_artifact"])
    if model_path.exists():
        try:
            import joblib
            model = joblib.load(model_path)
            # Expect a features parquet aligned to df index (Phase II)
            feat_path = Path(cfg["paths"]["features_parquet"])
            if feat_path.exists():
                feats = pd.read_parquet(feat_path)
                feats = feats.reindex(df.index).fillna(method="ffill").fillna(0.0)
                proba = model.predict_proba(feats)  # class order assumed ["DOWN","SIDEWAYS","UP"]
                labels = np.array(cfg["target"]["classes"])
                preds = labels[np.argmax(proba, axis=1)]
                return pd.Series(preds, index=df.index)
        except Exception:
            pass
    # Heuristic fallback
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    ema30 = df["close"].ewm(span=30, adjust=False).mean()
    atr_med = df["atr"].median()
    low_vol = df["atr"] < 1.5 * atr_med
    up = (ema10 > ema30) & low_vol
    down = (ema10 < ema30) & low_vol
    sig = pd.Series("SIDEWAYS", index=df.index)
    sig[up] = "UP"
    sig[down] = "DOWN"
    return sig


# --- START SNIPPET: ensure saving outputs & diagnostics ---
def _save_backtest_outputs_safely(df_out: pd.DataFrame, trades: pd.DataFrame, out_dir: str | Path):
    """
    Robustly save equity curve and trades CSVs and a NaN diagnostics file.
    Use this helper near the end of the backtest run to guarantee outputs are written.
    """
    out_path = Path(out_dir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[BACKTEST-SAVE] FAILED to create out_dir {out_path}: {e}")
        raise

    # Convert index to ISO timestamp column for CSV readability
    try:
        df_to_save = df_out.copy()
        # If index is datetime-like, move it to a column called 'timestamp'
        if hasattr(df_to_save.index, "tz") or hasattr(df_to_save.index, "tzinfo") or pd.api.types.is_datetime64_any_dtype(df_to_save.index):
            df_to_save = df_to_save.reset_index()
            df_to_save.rename(columns={df_to_save.columns[0]: "timestamp"}, inplace=True)
        else:
            # ensure there is a timestamp column
            if "timestamp" not in df_to_save.columns:
                df_to_save = df_to_save.reset_index().rename(columns={df_to_save.columns[0]: "timestamp"})

        # Exec price NaN handling: forward-fill then back-fill if necessary (do not mask results)
        if "exec_price" in df_to_save.columns and df_to_save["exec_price"].isna().any():
            print(f"[BACKTEST-SAVE] exec_price has NaNs: {df_to_save['exec_price'].isna().sum()} — will forward-fill then backfill before saving CSV.")
            df_to_save["exec_price"] = df_to_save["exec_price"].ffill().bfill()

        # Save equity curve CSV
        equity_csv = out_path / "equity_curve.csv"
        df_to_save.to_csv(equity_csv, index=False)
        print(f"[BACKTEST-SAVE] Saved equity curve to {equity_csv} ({len(df_to_save)} rows).")

        # Save NaN diagnostics if any NaNs remain
        nans = df_to_save[df_to_save.isna().any(axis=1)]
        if len(nans) > 0:
            diag_csv = out_path / "df_out_nans.csv"
            nans.to_csv(diag_csv, index=False)
            print(f"[BACKTEST-SAVE] NaN diagnostics saved to {diag_csv} ({len(nans)} NaN rows).")
        else:
            print("[BACKTEST-SAVE] No NaNs remain in df_out after fill operations.")

    except Exception as e:
        print("[BACKTEST-SAVE] Exception while saving equity curve:", e)
        traceback.print_exc()

    # Save trades (force timestamp column too)
    try:
        trades_to_save = trades.copy()
        if hasattr(trades_to_save.index, "tz") or pd.api.types.is_datetime64_any_dtype(trades_to_save.index):
            trades_to_save = trades_to_save.reset_index().rename(columns={trades_to_save.columns[0]: "timestamp"})
        trades_csv = out_path / "trades.csv"
        trades_to_save.to_csv(trades_csv, index=False)
        print(f"[BACKTEST-SAVE] Saved trades to {trades_csv} ({len(trades_to_save)} rows).")
    except Exception as e:
        print("[BACKTEST-SAVE] Exception while saving trades:", e)
        traceback.print_exc()
# --- END SNIPPET ---
