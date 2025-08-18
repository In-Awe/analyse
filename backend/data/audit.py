"""
backend/data/audit.py

DataAuditor: dataset validation, gap/duplicate detection, spike detection,
imputation strategies, statistical characterization (log returns, ADF),
ACF/PACF for returns and squared returns, intraday seasonality profiling,
and report export.
"""
from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
except Exception:
    adfuller = acf = pacf = None


class DataAuditor:
    def __init__(self, freq: str = "1T"):
        """
        freq: expected pandas frequency string for regular timestamps ("1T" = 1 minute)
        """
        self.freq = freq

    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.Series:
        """Return boolean mask of duplicated rows (first occurrence kept)."""
        subset = subset or ["timestamp", "open", "high", "low", "close", "volume"]
        return df.duplicated(subset=subset, keep="first")

    def detect_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect missing timestamps assuming df['timestamp'] is datetime and sorted.
        Returns a DataFrame with missing intervals (start, end, expected_count).
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        ts = pd.to_datetime(df["timestamp"]).sort_values().reset_index(drop=True)
        full_range = pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], freq=self.freq)
        missing = full_range.difference(ts)
        if len(missing) == 0:
            return pd.DataFrame(columns=["missing_ts"])
        return pd.DataFrame({"missing_ts": missing})

    def detect_spikes(self, df: pd.DataFrame, column: str = "close", window: int = 20, z_thresh: float = 8.0) -> pd.Series:
        """
        Simple spike detector using rolling median and rolling MAD-derived robust std.
        Returns boolean Series where True marks a spike.
        """
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        vals = pd.to_numeric(df[column], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        roll_med = vals.rolling(window=window, min_periods=1, center=False).median()
        # robust scale via MAD
        mad = (vals - roll_med).abs().rolling(window=window, min_periods=1).median()
        # approximate std from MAD: std ≈ 1.4826 * MAD
        roll_std = mad * 1.4826
        roll_std = roll_std.replace(0, np.nan).fillna(method="bfill").fillna(method="ffill").fillna(1e-8)
        z = (vals - roll_med).abs() / roll_std
        return z > z_thresh

    def impute(self, df: pd.DataFrame, method: str = "ffill", limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Impute missing numeric values in ['open','high','low','close','volume'].
        method: 'ffill', 'bfill', 'linear'
        Returns (df_imputed, is_imputed_mask)
        """
        numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df2 = df.copy()
        is_nan = df2[numeric_cols].isna().any(axis=1)
        if method == "linear":
            df2[numeric_cols] = df2[numeric_cols].interpolate(method="linear", limit=limit, limit_direction="both")
        elif method in ("ffill", "bfill"):
            df2[numeric_cols] = df2[numeric_cols].fillna(method=method, limit=limit)
        else:
            raise ValueError("unsupported imputation method")
        # for any remaining NaNs, fill with nearest value
        df2[numeric_cols] = df2[numeric_cols].fillna(method="ffill").fillna(method="bfill")
        is_imputed = is_nan & df2[numeric_cols].notna().any(axis=1)
        is_imputed = is_imputed.astype(bool)
        # add column
        df2["is_imputed_datapoint"] = is_imputed
        return df2, is_imputed

    def compute_log_returns(self, df: pd.DataFrame, col: str = "close", out_col: str = "log_return") -> pd.DataFrame:
        """Add log return column."""
        df2 = df.copy()
        if col not in df2.columns:
            df2[out_col] = np.nan
            return df2
        df2[out_col] = np.log(pd.to_numeric(df2[col], errors="coerce")).diff()
        return df2

    def adf_test(self, series: pd.Series) -> Dict[str, Any]:
        """Run Augmented Dickey-Fuller test; return results dict (or empty if statsmodels missing)."""
        if adfuller is None:
            return {"error": "statsmodels not installed"}
        series_clean = series.dropna()
        if len(series_clean) < 10:
            return {"error": "not enough data for ADF"}
        res = adfuller(series_clean, autolag="AIC")
        return {
            "adf_stat": float(res[0]),
            "pvalue": float(res[1]),
            "usedlag": int(res[2]),
            "nobs": int(res[3]),
            "crit_vals": {k: float(v) for k, v in res[4].items()},
            "icbest": float(res[5]),
        }

    def compute_acf_pacf(self, series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
        """Return ACF and PACF arrays (or error if statsmodels missing)."""
        if acf is None or pacf is None:
            return {"error": "statsmodels not installed"}
        s = series.dropna()
        if len(s) < 2:
            return {"acf": [], "pacf": []}
        a = acf(s, nlags=nlags, fft=True, missing="drop")
        p = pacf(s, nlags=nlags, method="ywunbiased")
        return {"acf": a.tolist(), "pacf": p.tolist()}

    def intraday_profile(self, df: pd.DataFrame, return_col: str = "log_return") -> pd.DataFrame:
        """
        Compute intraday seasonality profiles:
        - avg volume by hour/minute
        - avg absolute return (vol) by hour/minute
        Returns DataFrame indexed by hour,minute with columns avg_volume, avg_abs_ret.
        """
        df2 = df.copy()
        df2["timestamp"] = pd.to_datetime(df2["timestamp"])
        df2["hour"] = df2["timestamp"].dt.hour
        df2["minute"] = df2["timestamp"].dt.minute
        profile = (
            df2.groupby(["hour", "minute"])
            .agg(avg_volume=("volume", "mean"), avg_abs_ret=(return_col, lambda s: s.abs().mean()))
            .reset_index()
        )
        return profile

    def audit(self, df: pd.DataFrame, impute_method: Optional[str] = "ffill", impute_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a full audit and return a report dict. This does not mutate the input df.
        """
        report: Dict[str, Any] = {}
        if "timestamp" in df.columns:
            report["n_rows"] = int(len(df))
            # duplicates
            dup_mask = self.detect_duplicates(df)
            report["n_duplicates"] = int(dup_mask.sum())
            report["duplicate_sample"] = df[dup_mask].head(5).to_dict("records")
            # gaps
            gaps = self.detect_gaps(df)
            report["n_missing_timestamps"] = int(len(gaps))
            if len(gaps) > 0:
                report["missing_sample"] = gaps.head(5).to_dict("records")
        else:
            report["warning"] = "no timestamp column"

        # spike detection on close/open/high/low
        spikes = {}
        for c in ["close", "open", "high", "low"]:
            if c in df.columns:
                mask = self.detect_spikes(df, column=c)
                spikes[c] = {"n_spikes": int(mask.sum()), "sample_idx": df.index[mask].tolist()[:5]}
        report["spikes"] = spikes

        # impute
        df_imputed, is_imputed = self.impute(df, method=impute_method or "ffill", limit=impute_limit)
        report["n_imputed_points"] = int(is_imputed.sum())
        report["impute_method"] = impute_method

        # compute log returns & adf
        df_lr = self.compute_log_returns(df_imputed)
        report["log_return_na"] = int(df_lr["log_return"].isna().sum())
        report["adf_log_return"] = self.adf_test(df_lr["log_return"])

        # acf/pacf for returns and squared returns
        report["acf_pacf_log_return"] = self.compute_acf_pacf(df_lr["log_return"])
        report["acf_pacf_sq_log_return"] = self.compute_acf_pacf(df_lr["log_return"].dropna() ** 2)

        # intraday profile
        try:
            profile = self.intraday_profile(df_lr, return_col="log_return")
            # include a small summary
            top_vol = profile.sort_values("avg_volume", ascending=False).head(3).to_dict("records")
            top_vol_hours = [f"{r['hour']:02d}:{r['minute']:02d}" for r in top_vol]
            report["intraday_top_volume_slots"] = top_vol_hours
            report["intraday_profile_sample"] = profile.head(5).to_dict("records")
        except Exception as e:
            report["intraday_error"] = str(e)

        # return full report and also the imputed dataframe for downstream use
        return {"report": report, "df_imputed": df_imputed, "is_imputed_mask": is_imputed}

    def export_report(self, report: Dict[str, Any], out_dir: str = "backend/reports") -> str:
        """
        Save a minimal report to out_dir as JSON and return path.
        (User can inspect the raw report dict as needed.)
        """
        import json

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "data_audit_report.json")
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        return path

if __name__ == "__main__":  # quick demo
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="backend/reports")
    p.add_argument("--freq", default="1T")
    args = p.parse_args()
    csv = Path(args.csv)
    df = pd.read_csv(csv)
    auditor = DataAuditor(freq=args.freq)
    result = auditor.audit(df)
    path = auditor.export_report(result["report"], out_dir=args.out)
    print("Saved report to", path)
