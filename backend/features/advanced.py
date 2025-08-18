"""
backend/features/advanced.py

Vectorized implementations of:
- VWAP (rolling)
- ATR
- Bollinger Bands
- MACD (+signal)
- OBV
- VVR (Volume-to-Volatility Ratio)
- volatility-of-volatility
- candlestick morphology (body/total, upper/lower wick ratios)
- microstructure proxies (rolling price-volume corr, ADV)
- multi-timescale EMAs
- interaction features (e.g., RSI * ATR) when inputs available

Also includes build_feature_matrix and feature_report helpers.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb  # optional: used for importance ranking if available
except Exception:
    xgb = None


def rolling_vwap(df: pd.DataFrame, window: int = 20, price_col: str = "close", vol_col: str = "volume", out_col: str = "vwap") -> pd.DataFrame:
    """Rolling VWAP (windowed)."""
    df = df.copy()
    if price_col not in df.columns or vol_col not in df.columns:
        df[out_col] = pd.NA
        return df
    tp = df[price_col] * df[vol_col]
    num = tp.rolling(window=window, min_periods=1).sum()
    den = df[vol_col].rolling(window=window, min_periods=1).sum().replace(0, np.nan).fillna(method="bfill").fillna(method="ffill")
    df[out_col] = num / den
    return df


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range series."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14, out_col: str = "atr") -> pd.DataFrame:
    df = df.copy()
    df[out_col] = true_range(df).rolling(window=window, min_periods=1).mean()
    return df


def bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0, price_col: str = "close", prefix: str = "bb") -> pd.DataFrame:
    df = df.copy()
    ma = df[price_col].rolling(window=window, min_periods=1).mean()
    sd = df[price_col].rolling(window=window, min_periods=1).std()
    df[f"{prefix}_mid"] = ma
    df[f"{prefix}_upper"] = ma + n_std * sd
    df[f"{prefix}_lower"] = ma - n_std * sd
    df[f"{prefix}_width"] = (df[f"{prefix}_upper"] - df[f"{prefix}_lower"]) / df[f"{prefix}_mid"].replace(0, np.nan)
    return df


def ema(df: pd.DataFrame, span: int = 12, col: str = "close", out_col: Optional[str] = None) -> pd.Series:
    out_col = out_col or f"ema_{span}"
    return df[col].ewm(span=span, adjust=False).mean().rename(out_col)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "close") -> pd.DataFrame:
    df = df.copy()
    ema_fast = ema(df, span=fast, col=price_col)
    ema_slow = ema(df, span=slow, col=price_col)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def obv(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume", out_col: str = "obv") -> pd.DataFrame:
    df = df.copy()
    sign = np.sign(df[price_col].diff().fillna(0))
    df[out_col] = (sign * df[vol_col]).fillna(0).cumsum()
    return df


def vvr(df: pd.DataFrame, vol_window: int = 20, ret_col: str = "log_return", vol_col: str = "volume", out_col: str = "vvr") -> pd.DataFrame:
    """Volume-to-Volatility Ratio: rolling volume / rolling volatility of returns"""
    df = df.copy()
    if ret_col not in df.columns or vol_col not in df.columns:
        df[out_col] = pd.NA
        return df
    vol = df[ret_col].rolling(window=vol_window, min_periods=1).std().replace(0, np.nan).fillna(method="bfill").fillna(method="ffill")
    vol_sum = df[vol_col].rolling(window=vol_window, min_periods=1).mean()
    df[out_col] = vol_sum / (vol + 1e-12)
    return df


def vol_of_vol(df: pd.DataFrame, ret_col: str = "log_return", window: int = 20, out_col: str = "vol_of_vol") -> pd.DataFrame:
    df = df.copy()
    rolling_std = df[ret_col].rolling(window=window, min_periods=1).std()
    df[out_col] = rolling_std.rolling(window=window, min_periods=1).std()
    return df


def candle_morphology(df: pd.DataFrame, prefix: str = "candle") -> pd.DataFrame:
    """
    Compute candle ratios:
      - body_len / total_len
      - upper_wick / total_len
      - lower_wick / total_len
    """
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    total = (df["high"] - df["low"]).replace(0, np.nan)
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    df[f"{prefix}_body_ratio"] = body / total
    df[f"{prefix}_upper_wick_ratio"] = upper / total
    df[f"{prefix}_lower_wick_ratio"] = lower / total
    df[f"{prefix}_body_signed"] = (df["close"] - df["open"]) / (total + 1e-12)
    df.fillna(0, inplace=True)
    return df


def rolling_price_volume_corr(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume", window: int = 20, out_col: str = "pv_corr") -> pd.DataFrame:
    df = df.copy()
    # correlate absolute returns with volume
    df["abs_ret"] = df[price_col].pct_change().abs()
    df[out_col] = df["abs_ret"].rolling(window=window, min_periods=1).corr(df[vol_col])
    df.drop(columns=["abs_ret"], inplace=True)
    return df


def adv(df: pd.DataFrame, window: int = 1440, vol_col: str = "volume", out_col: str = "adv") -> pd.DataFrame:
    """Average daily volume approximated by rolling window (default 1440 minutes)."""
    df = df.copy()
    df[out_col] = df[vol_col].rolling(window=window, min_periods=1).mean()
    return df


def multi_ema(df: pd.DataFrame, spans: List[int], price_col: str = "close") -> pd.DataFrame:
    df = df.copy()
    for s in spans:
        df[f"ema_{s}"] = ema(df, span=s, col=price_col)
    return df


def interaction_features(df: pd.DataFrame, left_cols: List[str], right_cols: List[str]) -> pd.DataFrame:
    """Create pairwise interaction columns left * right if both present."""
    df = df.copy()
    for l in left_cols:
        for r in right_cols:
            if l in df.columns and r in df.columns:
                name = f"{l}_x_{r}"
                df[name] = df[l] * df[r]
    return df


def build_feature_matrix(df: pd.DataFrame, compute_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build a wide feature DataFrame from available columns.
    compute_list: list of features to compute; if None compute defaults.
    """
    compute_list = compute_list or [
        "ema_multi",
        "vwap_20",
        "atr_14",
        "bb_20",
        "macd",
        "obv",
        "vvr_20",
        "vol_of_vol_20",
        "candles",
        "pv_corr_20",
        "adv_1440",
    ]
    out = df.copy()
    # ensure log_return exists if possible
    if "log_return" not in out.columns and "close" in out.columns:
        out["log_return"] = np.log(out["close"]).diff()

    if "ema_multi" in compute_list:
        out = multi_ema(out, spans=[5, 10, 20, 50, 100, 200])
    if "vwap_20" in compute_list:
        out = rolling_vwap(out, window=20)
    if "atr_14" in compute_list:
        out = atr(out, window=14)
    if "bb_20" in compute_list:
        out = bollinger_bands(out, window=20)
    if "macd" in compute_list:
        out = macd(out)
    if "obv" in compute_list:
        out = obv(out)
    if "vvr_20" in compute_list:
        out = vvr(out, vol_window=20)
    if "vol_of_vol_20" in compute_list:
        out = vol_of_vol(out, ret_col="log_return", window=20)
    if "candles" in compute_list:
        out = candle_morphology(out)
    if "pv_corr_20" in compute_list:
        out = rolling_price_volume_corr(out, window=20)
    if "adv_1440" in compute_list:
        out = adv(out, window=1440)

    # interaction features example: RSI (if present) x ATR
    left = [c for c in ["rsi", "atr", "vvr", "adv"] if c in out.columns]
    right = [c for c in ["ema_20", "ema_50", "vol_of_vol"] if c in out.columns]
    out = interaction_features(out, left_cols=left, right_cols=right)

    # drop intermediate columns that are clearly non-feature (like timestamp)
    # keep timestamp if present
    return out


def feature_report(df: pd.DataFrame, target_col: Optional[str] = None, out_dir: str = "backend/reports") -> dict:
    """
    Compute correlation matrix and feature importances.
    If target_col provided, attempt to compute feature importances with XGBoost or RandomForest.
    Returns dict with keys: correlation (DataFrame), importances (DataFrame or dict)
    """
    os.makedirs(out_dir, exist_ok=True)
    features = df.select_dtypes(include=[np.number]).copy()
    if features.empty:
        return {"error": "no numeric features"}

    corr = features.corr()
    corr_path = os.path.join(out_dir, "feature_correlation.csv")
    corr.to_csv(corr_path)

    importances = None
    if target_col and target_col in df.columns:
        # prepare supervised target: if continuous, discretize to 3 classes (down/side/up) as example
        y = df[target_col].dropna()
        X = features.loc[y.index].fillna(0)
        if len(y.unique()) > 10 and np.issubdtype(y.dtype, np.floating):
            # discretize by quantiles to three classes
            y_bins = pd.qcut(y, q=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2], duplicates="drop")
            y_used = y_bins.astype(int)
        else:
            y_used = y

        # try XGBoost first
        try:
            if xgb is not None:
                dtrain = xgb.DMatrix(X, label=y_used)
                params = {"objective": "multi:softprob" if len(np.unique(y_used)) > 2 else "binary:logistic", "verbosity": 0}
                num_class = len(np.unique(y_used))
                if num_class > 2:
                    params["num_class"] = num_class
                bst = xgb.train(params, dtrain, num_boost_round=50)
                imp = bst.get_score(importance_type="gain")
                imp_df = pd.DataFrame(list(imp.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False)
                importances = imp_df
            else:
                raise ImportError("xgboost not installed")
        except Exception:
            # fallback to RandomForest
            rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
            # if multiclass label may be categorical
            try:
                rf.fit(X, y_used)
                imp = rf.feature_importances_
                imp_df = pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values("importance", ascending=False)
                importances = imp_df
            except Exception as e:
                importances = {"error": str(e)}

        # save importances
        try:
            if isinstance(importances, pd.DataFrame):
                importances.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
        except Exception:
            pass

    return {"correlation_csv": corr_path, "importances": importances}


if __name__ == "__main__":
    # quick smoke test
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="backend/reports")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    features = build_feature_matrix(df)
    report = feature_report(features, target_col="log_return", out_dir=args.out)
    print("Saved correlation to", report.get("correlation_csv"))
    print("Importances head:", getattr(report.get("importances"), "head", lambda: report.get("importances"))())
