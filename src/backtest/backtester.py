#!/usr/bin/env python3
"""
Vectorized backtester scaffold for Phase IV.

Key behaviors:
 - Accepts merged features + target parquet (must include timestamp, open, close, ATR columns if using ATR slippage)
 - Produces signals either by loading a saved XGBoost model (JSON) or by reading a signal column
 - Executes trades at next bar open (execution_delay_bars) and applies slippage and fees
 - Vectorized P&L calculation based on bar returns and position fraction allocation
 - Produces trades DataFrame, equity curve, returns series, summary metrics (via src/backtest/metrics.py)

This is a scaffold — it is conservative, clear, and designed to be iterated upon.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from src.backtest.metrics import summary_from_returns
from src.backtest import model_loaders
import math

# helper
def load_config(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_merged_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # ensure timestamp sorted and datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def load_xgb_model(model_path: str) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    return bst

def predict_with_xgb(bst: xgb.Booster, X: pd.DataFrame, meta: Dict[str, Any], predict_proba: bool=True) -> np.ndarray:
    # meta expected to contain 'feature_names' and 'enc_to_label' mapping
    feat_names = meta.get("feature_names", X.columns.tolist())
    # reindex X to feature_names safely
    Xm = X.reindex(columns=feat_names).fillna(0.0)
    dmat = xgb.DMatrix(Xm.values, feature_names=feat_names)
    preds = bst.predict(dmat)
    if preds.ndim == 2:
        # multi-class probabilities
        if predict_proba:
            return preds  # shape (n_samples, n_classes)
        else:
            return np.argmax(preds, axis=1)
    else:
        # binary prob returned as single-column
        if predict_proba:
            return np.vstack([1 - preds, preds]).T
        else:
            return (preds > 0.5).astype(int)

def decode_preds(preds_enc: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    # meta.enc_to_label expected e.g. {"0": -1, "1": 0, "2": 1}
    enc_to_label = meta.get("enc_to_label") or meta.get("enc_to_label", {})
    # ensure keys are strings
    decoded = []
    for p in preds_enc:
        key = str(int(p))
        decoded.append(int(enc_to_label.get(key, int(p))))
    return np.array(decoded, dtype=int)

def compute_slippage_percent(row: pd.Series, cfg: Dict) -> float:
    st = cfg["execution"]["slippage"]["type"]
    if st == "fixed":
        return float(cfg["execution"]["slippage"]["fixed_pct"])
    # atr_fraction: slippage_pct = (atr_next / open_next) * atr_fraction
    if st == "atr_fraction":
        # use the atr column if present; else fall back to computed approx
        atr_col = "atr_14"  # common name from feature builder
        if atr_col in row:
            # atr value is in price units, convert to percentage vs open
            open_p = row.get("open", np.nan)
            if open_p and open_p > 0:
                return float((row.get(atr_col, 0.0) / open_p) * cfg["execution"]["slippage"]["atr_fraction"])
        # fallback
        return float(cfg["pricing"].get("slippage_alpha", 0.5) * cfg["execution"].get("slippage", {}).get("fixed_pct", 0.0005))
    return 0.0

def _construct_signals(df: pd.DataFrame, cfg: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      - signal_labels: array of -1/0/1 (length = len(df))
      - signal_probs: if model proba available, returns per-class probs (n x k) else None
    """
    source = cfg["signals"]["source"]
    if source == "col":
        col = cfg["signals"]["col_name"]
        if col not in df.columns:
            raise ValueError(f"signal column {col} not present in dataframe")
        sig = df[col].fillna(0).astype(int).values
        return sig, None
    elif source == "model":
        model_cfg = cfg["signals"]["model"]
        mtype = model_cfg.get("type", "xgboost")
        # prepare feature-only matrix for model inputs (drop obvious price/time cols)
        X = df.select_dtypes(include=[np.number]).copy()
        for dropc in ["open", "high", "low", "close", "volume", "timestamp"]:
            if dropc in X.columns:
                X = X.drop(columns=[dropc])
        if mtype == "xgboost":
            model_path = model_cfg["path"]
            meta_path = model_cfg.get("meta")
            bst, meta = model_loaders.load_xgb_model_and_meta(model_path, meta_path)
            labels_enc, probs = model_loaders.predict_xgb(bst, X, meta, predict_proba=model_cfg.get("predict_proba", True))
            # decode using meta.enc_to_label if present
            enc_to_label = meta.get("enc_to_label", {})
            if enc_to_label:
                labels = np.array([int(enc_to_label.get(str(int(v)), int(v))) for v in labels_enc], dtype=int)
            else:
                labels = labels_enc
            return labels, probs
        elif mtype == "lstm":
            # LSTM: require paths in config.models.lstm
            lstm_cfg = cfg.get("models", {}).get("lstm", {})
            model_path = lstm_cfg.get("model_path") or model_cfg.get("path")
            scaler_path = lstm_cfg.get("scaler_path")
            label_map = lstm_cfg.get("label_map")
            seq_len = int(lstm_cfg.get("seq_len", 60))
            batch_size = int(lstm_cfg.get("batch_size", 512))
            art = model_loaders.load_lstm_artifact(model_path, scaler_path, label_map, device="cpu")
            # pass numeric X into LSTM predictor
            labels, probs = model_loaders.predict_lstm(art, X, seq_len=seq_len, batch_size=batch_size)
            return labels, probs
        else:
            raise ValueError(f"unsupported model type for signals: {mtype}")
    else:
        raise ValueError("unknown signals.source config")

def vectorized_backtest(df: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Core vectorized backtest logic.

    Returns a dict with:
      - equity_curve (pd.Series indexed by timestamp)
      - returns_series (np.array)
      - trades (pd.DataFrame)
      - summary (dict)
    """
    # basic params
    capital = float(cfg["capital"]["initial_capital"])
    position_fraction = float(cfg["capital"]["position_fraction"])
    exec_delay = int(cfg["execution"]["execution_delay_bars"])
    maker_fee = float(cfg["execution"]["fee"]["maker_fee_pct"])
    taker_fee = float(cfg["execution"]["fee"]["taker_fee_pct"])
    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy().reset_index(drop=True)
    n = len(df)
    # build signals
    signal_labels, signal_probs = _construct_signals(df, cfg)
    df["signal"] = signal_labels

    # positions are fraction of equity allocated with sign (-1/0/1)
    pos_frac = (signal_labels.astype(float) * position_fraction)
    # applied position after execution delay (shift forward)
    applied_pos = np.concatenate([np.zeros(exec_delay), pos_frac[:-exec_delay]]) if exec_delay > 0 else pos_frac.copy()

    # compute bar returns (open -> close) as fractional returns
    open_p = df[cfg["input"]["price_column_open"]].values
    close_p = df[cfg["input"]["price_column_close"]].values
    # bar return: (close / open) - 1, avoid divide by zero
    bar_ret = np.where(open_p > 0, (close_p / open_p) - 1.0, 0.0)

    # slippage percent per bar (uses next bar atr/open if possible). We'll compute per-row slippage_pct
    slippage_pct = np.zeros(n)
    for i in range(n):
        slippage_pct[i] = compute_slippage_percent(df.iloc[i].to_dict() if hasattr(df.iloc[i], "to_dict") else df.iloc[i], cfg)

    # trade events where applied_pos changes
    applied_pos_series = pd.Series(applied_pos)
    pos_change = applied_pos_series.diff().fillna(applied_pos_series).values != 0

    # trade cost vector: when a trade occurs (entry or exit), apply taker fee on notional = position_fraction * price
    # approximate monetary cost per bar relative to equity = fee_pct * position_fraction
    trade_cost_pct = np.zeros(n)
    for i in range(n):
        if pos_change[i]:
            # determine if entering or exiting; apply taker_fee for both entry and exit events (conservative)
            trade_cost_pct[i] = taker_fee * abs(applied_pos[i]) / (position_fraction if position_fraction>0 else 1.0)
        else:
            trade_cost_pct[i] = 0.0

    # slippage cost as percent of price applied at entries as well (approx)
    slippage_cost_pct = pos_change * slippage_pct

    # strategy returns per bar as fraction of equity:
    # strategy_return_pct = applied_pos * bar_ret - trade_cost_pct - slippage_cost_pct
    strat_ret = applied_pos * bar_ret - trade_cost_pct - slippage_cost_pct

    # equity curve
    equity = [capital]
    for r in strat_ret:
        equity.append(equity[-1] * (1.0 + float(r)))
    equity = pd.Series(equity[1:], index=df["timestamp"])

    # build trades DataFrame (iterative over pos changes)
    trades = []
    current_entry = None
    for i in range(n):
        if pos_change[i]:
            ts = df.loc[i, "timestamp"]
            sign = np.sign(applied_pos[i])
            price = open_p[i]  # approximate execution at open
            slippage = slippage_pct[i] * price
            exec_price = price + (slippage * sign)
            if current_entry is None and sign != 0:
                # entry
                current_entry = {
                    "entry_index": i,
                    "entry_time": ts,
                    "side": int(sign),
                    "entry_price": float(exec_price),
                    "size_fraction": abs(applied_pos[i]),
                }
            elif current_entry is not None:
                # exit existing trade, record
                exit_price = exec_price
                pnl = (exit_price - current_entry["entry_price"]) * current_entry["side"] * (capital * current_entry["size_fraction"] / current_entry["entry_price"])
                trades.append({
                    "entry_time": current_entry["entry_time"],
                    "exit_time": ts,
                    "side": current_entry["side"],
                    "entry_price": current_entry["entry_price"],
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "size_fraction": current_entry["size_fraction"],
                })
                # if new sign is non-zero, start new entry
                if sign != 0:
                    current_entry = {
                        "entry_index": i,
                        "entry_time": ts,
                        "side": int(sign),
                        "entry_price": float(exec_price),
                        "size_fraction": abs(applied_pos[i]),
                    }
                else:
                    current_entry = None
    # if a position still open at end, close at last bar close
    if current_entry is not None:
        ts = df.loc[n-1, "timestamp"]
        price = close_p[-1]
        sign = current_entry["side"]
        exit_price = price
        pnl = (exit_price - current_entry["entry_price"]) * current_entry["side"] * (capital * current_entry["size_fraction"] / current_entry["entry_price"])
        trades.append({
            "entry_time": current_entry["entry_time"],
            "exit_time": ts,
            "side": current_entry["side"],
            "entry_price": current_entry["entry_price"],
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "size_fraction": current_entry["size_fraction"],
        })

    trades_df = pd.DataFrame(trades)

    # compute returns series as percent returns over periods from equity
    returns = equity.pct_change().fillna(0.0).values

    # metrics
    # period_per_year: use minutes-per-year (approx)
    period_per_year = 365 * 24 * 60
    summary = summary_from_returns(equity, returns, trades_df, period_per_year)

    # save artifacts if requested
    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["outputs"].get("save_equity_csv", True):
        eq_df = pd.DataFrame({"timestamp": equity.index, "equity": equity.values})
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)
    if cfg["outputs"].get("save_trades_csv", True):
        trades_df.to_csv(out_dir / "trades.csv", index=False)
    if cfg["outputs"].get("save_summary_json", True):
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return {
        "equity": equity,
        "returns": returns,
        "trades": trades_df,
        "summary": summary,
    }

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/backtest.yaml", help="backtest config yaml")
    p.add_argument("--merged", default=None, help="override merged features parquet")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.merged:
        cfg["backtest"]["input"]["merged_features"] = args.merged
    merged_path = cfg["backtest"]["input"]["merged_features"]
    df = load_merged_features(merged_path)
    # transfer cfg into easier structure (backtest.* -> top-level)
    # flatten names for backward compat
    flat_cfg = {}
    flat_cfg.update(cfg["backtest"])
    flat_cfg.update(cfg.get("backtest", {}))
    # include other top-level nodes
    flat_cfg["execution"] = cfg.get("backtest", {}).get("execution", cfg.get("execution", cfg.get("backtest", {}).get("execution", {})))
    # add signals and capital info
    # we support both 'signals' under config root and under backtest
    if "signals" not in flat_cfg:
        flat_cfg["signals"] = cfg.get("backtest", {}).get("signals", cfg.get("signals", {}))
    if "capital" not in flat_cfg:
        flat_cfg["capital"] = cfg.get("backtest", {}).get("capital", cfg.get("capital", {}))
    # set outputs
    flat_cfg["outputs"] = cfg.get("backtest", {}).get("outputs", cfg.get("outputs", {}))
    # pass df and flattened cfg
    res = vectorized_backtest(df, flat_cfg)
    print("[backtest] summary:", json.dumps(res["summary"], indent=2))
    print(f"[backtest] saved artifacts to {flat_cfg['outputs']['out_dir']}")

if __name__ == "__main__":
    main()
