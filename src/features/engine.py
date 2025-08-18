#!/usr/bin/env python3
"""
Feature engineering engine for Phase II.
Reads cleaned OHLCV CSV (expected columns: timestamp, open, high, low, close, volume,
and optional flags is_imputed, is_spike, spike_magnitude).
Produces a parquet file with technical + microstructure + interaction features.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

EPS = 1e-12

def read_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = df.sort_index().reset_index()
    # ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def macd(close: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})

def rsi(close: pd.Series, window: int=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + EPS)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.fillna(0.0)

def atr(df: pd.DataFrame, window: int=14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=window, adjust=False).mean()

def bollinger_bands(close: pd.Series, window: int=20, ndev: int=2):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std().fillna(0)
    upper = ma + ndev * sd
    lower = ma - ndev * sd
    return ma, upper, lower

def vwap(df: pd.DataFrame, window: int) -> pd.Series:
    # rolling VWAP: sum(price*volume)/sum(volume) over window
    pv = (df["close"] * df["volume"]).rolling(window).sum()
    v = df["volume"].rolling(window).sum()
    return (pv / (v + EPS)).fillna(method="ffill")

def vvr(df: pd.DataFrame, vol_window: int=30) -> pd.Series:
    # volume / rolling_std(returns)
    ret = np.log(df["close"]).diff().fillna(0)
    vol = ret.rolling(vol_window).std()
    return df["volume"] / (vol + EPS)

def vol_of_vol(df: pd.DataFrame, window: int=120) -> pd.Series:
    a = atr(df, 14)
    return a.rolling(window).std().fillna(0)

def candle_morphology(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    total = (df["high"] - df["low"]).replace(0, EPS)
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    return pd.DataFrame({
        "body_ratio": (body / total).fillna(0.0),
        "upper_wick_ratio": (upper_wick / total).fillna(0.0),
        "lower_wick_ratio": (lower_wick / total).fillna(0.0),
    })

def rolling_corr(a: pd.Series, b: pd.Series, window: int=60) -> pd.Series:
    return a.rolling(window).corr(b).fillna(0.0)

def build_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    # technicals
    for p in cfg.get("technical_indicators", {}).get("sma", []):
        out[f"sma_{p}"] = sma(out["close"], p)
    for p in cfg.get("technical_indicators", {}).get("ema", []):
        out[f"ema_{p}"] = ema(out["close"], p)
    for mac in cfg.get("technical_indicators", {}).get("macd", []):
        mdf = macd(out["close"], fast=mac["fast"], slow=mac["slow"], signal=mac["signal"])
        out[f"macd_{mac['fast']}_{mac['slow']}"] = mdf["macd"]
        out[f"macd_signal_{mac['fast']}_{mac['slow']}"] = mdf["macd_signal"]
        out[f"macd_hist_{mac['fast']}_{mac['slow']}"] = mdf["macd_hist"]
    for p in cfg.get("technical_indicators", {}).get("rsi", []):
        out[f"rsi_{p}"] = rsi(out["close"], p)
    for p in cfg.get("technical_indicators", {}).get("atr", []):
        out[f"atr_{p}"] = atr(out, p)
    for boll in cfg.get("technical_indicators", {}).get("bollinger", []):
        ma, upper, lower = bollinger_bands(out["close"], window=boll["window"], ndev=boll["ndev"])
        out[f"bb_ma_{boll['window']}"] = ma
        out[f"bb_upper_{boll['window']}"] = upper
        out[f"bb_lower_{boll['window']}"] = lower

    # microstructure
    for w in cfg.get("microstructure", {}).get("vwap", {}).get("windows", []):
        out[f"vwap_{w}"] = vwap(out, window=w)
    vvr_cfg = cfg.get("microstructure", {}).get("vvr", {})
    out[f"vvr_{vvr_cfg.get('window',30)}"] = vvr(out, vol_window=vvr_cfg.get("window", 30))
    out["vol_of_vol"] = vol_of_vol(out, cfg.get("microstructure", {}).get("vol_of_vol", {}).get("window", 120))
    if cfg.get("microstructure", {}).get("candle_morphology", True):
        morph = candle_morphology(out)
        for c in morph.columns:
            out[c] = morph[c]

    # price-volume correlation
    pvcfg = cfg.get("microstructure", {}).get("vol_price_corr", {})
    out[f"vol_price_corr_{pvcfg.get('window',60)}"] = rolling_corr(out["volume"], (np.log(out["close"]).diff().abs().fillna(0)), window=pvcfg.get("window",60))

    # interactions
    if cfg.get("interaction_features", {}).get("enabled", True):
        combos = cfg.get("interaction_features", {}).get("combos", [])
        for combo in combos:
            names = []
            for key in combo:
                names.append(key)
            # safe multiply: if column missing, fill 0
            cols = [out[c] if c in out.columns else 0.0 for c in names]
            newname = "_x_".join(names)
            out[newname] = pd.concat(cols, axis=1).prod(axis=1)

    # bring forward useful flags if present
    for flag in ["is_imputed", "is_spike", "spike_magnitude"]:
        if flag in df.columns:
            out[flag] = df[flag]

    # final housekeeping: drop rows with NaN for close (if any)
    out = out.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
    return out

def load_config(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="cleaned CSV input file")
    p.add_argument("--config", default="configs/features.yaml", help="features config yaml")
    p.add_argument("--out", default="data/features/technical.parquet", help="output parquet")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    df = read_clean_csv(args.input)
    feats = build_features(df, cfg)
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(outpath, index=False)
    print(f"Written features to {outpath} (rows: {len(feats)})")

if __name__ == "__main__":
    main()
