From 1111111111111111111111111111111111111111 Mon Sep 17 00:00:00 2001
From: repo-user <user@example.com>
Date: Mon, 18 Aug 2025 00:00:00 +0000
Subject: [PATCH] Improve core/data_ingest.py: CLI, robust column detection, config flags, logging, atomic outputs

---
 core/data_ingest.py | 288 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 288 insertions(+)
 create mode 100644 core/data_ingest.py
@@
-"""
-Data ingestion + integrity audit for BTCUSDT_1m_December_2024.csv
-
-This script performs the initial data quality checks as specified in Phase I 
-of the HFT system development directive. It addresses:
-1.  Duplicate timestamp detection and removal.
-2.  Identification of missing timestamps (gaps) in the 1-minute series.
-3.  Imputation of missing data using a forward-fill/backward-fill strategy.
-4.  Anomalous spike detection using a rolling Z-score.
-5.  Generation of two new features: `is_imputed_datapoint` and `spike_magnitude`.
-6.  A preliminary statistical analysis of log returns, including an ADF test.
-
-Outputs:
- - data/clean/BTCUSDT_1m_December_2024.cleaned.csv
- - data/reports/BTCUSDT_1m_December_2024_audit.json
-"""
-import os
-import json
-from datetime import timedelta
-import numpy as np
-import pandas as pd
-from statsmodels.tsa.stattools import adfuller
-
-# --- CONFIGURATION ---
-# Adjusted paths and column names to match the provided dataset.
-RAW_CSV = "BTCUSDT_1m_December_2024.csv"
-CLEAN_CSV = "data/clean/BTCUSDT_1m_December_2024.cleaned.csv"
-REPORT_JSON = "data/reports/BTCUSDT_1m_December_2024_audit.json"
-TIMESTAMP_COL = "datetime_utc" # MODIFIED: Column name in the source CSV is 'datetime_utc'
-TIMEZONE = "UTC"
-FREQ = "1T"  # pandas frequency string for 1 minute
-SPIKE_ZSCORE_THRESHOLD = 8.0  # Z-score threshold for flagging extreme price spikes
-
-# --- SCRIPT EXECUTION ---
-
-def setup_directories():
-    """Create output directories if they don't exist."""
-    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
-    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
-
-def read_and_prepare_data(path):
-    """
-    Reads the raw CSV, standardizes the timestamp column, and sorts the data.
-    """
-    print(f"Reading raw data from {path}...")
-    df = pd.read_csv(path)
-    
-    # Coerce the timestamp column to datetime objects with UTC timezone.
-    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
-    
-    # Sort by timestamp to ensure correct chronological order.
-    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
-    print(f"Successfully read {len(df)} rows.")
-    return df
-
-def audit_and_clean(df):
-    """
-    Performs the main data audit and cleaning pipeline.
-    """
-    print("Starting data audit and cleaning process...")
-    report = {}
-
-    # 1. Handle Duplicates
-    report['rows_raw'] = int(len(df))
-    dup_mask = df.duplicated(subset=[TIMESTAMP_COL], keep=False)
-    report['duplicate_timestamp_count'] = int(dup_mask.sum())
-    if report['duplicate_timestamp_count'] > 0:
-        # If duplicates exist, we keep the last recorded entry for that timestamp.
-        df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep='last').reset_index(drop=True)
-        print(f"Removed {report['duplicate_timestamp_count']} duplicate rows.")
-
-    # 2. Identify and Fill Gaps
-    df = df.set_index(TIMESTAMP_COL)
-    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FREQ, tz=TIMEZONE)
-    
-    report['first_timestamp'] = str(df.index.min())
-    report['last_timestamp'] = str(df.index.max())
-    report['expected_rows_for_timespan'] = int(len(full_idx))
-    report['missing_timestamps_count'] = int(len(full_idx.difference(df.index)))
-
-    # Reindex the dataframe to the full, gapless index and mark imputed points.
-    df_reindexed = df.reindex(full_idx)
-    df_reindexed['is_imputed'] = df_reindexed['close'].isna()
-    
-    # Use forward-fill then backward-fill to handle missing OHLCV data.
-    # This is a common approach to preserve continuity.
-    fill_cols = ['open', 'high', 'low', 'close', 'volume']
-    df_ffill = df_reindexed.copy()
-    df_ffill[fill_cols] = df_ffill[fill_cols].ffill().bfill()
-    print(f"Filled {report['missing_timestamps_count']} missing timestamps.")
-
-    # 3. Detect Spikes (Outliers)
-    # We use a rolling z-score to detect anomalous price movements relative to recent history.
-    rolling_window = 60 # 1 hour window
-    roll_mean = df_ffill['close'].rolling(window=rolling_window, min_periods=10).mean()
-    roll_std = df_ffill['close'].rolling(window=rolling_window, min_periods=10).std().replace(0, np.nan)
-    zscore = (df_ffill['close'] - roll_mean) / roll_std
-    
-    spike_mask = (zscore.abs() > SPIKE_ZSCORE_THRESHOLD) & df_ffill['close'].notna()
-    report['spike_candidates_found'] = int(spike_mask.sum())
-    df_ffill['spike_zscore'] = zscore
-    df_ffill['spike_candidate'] = spike_mask
-    print(f"Identified {report['spike_candidates_found']} potential price spikes.")
-
-    # 4. Feature Engineering (as per directive)
-    df_ffill['is_imputed_datapoint'] = df_ffill['is_imputed'].astype(int)
-    df_ffill['spike_magnitude'] = df_ffill['spike_zscore'].fillna(0).abs()
-
-    # 5. Calculate Log Returns for Statistical Analysis
-    df_ffill['log_return_1m'] = np.log(df_ffill['close'] / df_ffill['close'].shift(1))
-    df_ffill['log_return_1m'] = df_ffill['log_return_1m'].fillna(0)
-
-    # 6. Basic Distributional Statistics
-    lr = df_ffill['log_return_1m'].replace([np.inf, -np.inf], np.nan).dropna()
-    report['logreturn_mean'] = float(lr.mean())
-    report['logreturn_std'] = float(lr.std())
-    report['logreturn_skew'] = float(lr.skew())
-    report['logreturn_kurtosis'] = float(lr.kurtosis())
-
-    # 7. ADF Stationarity Test for Returns
-    try:
-        adf_res = adfuller(lr, autolag='AIC')
-        report['adf_stationarity_test'] = {
-            'statistic': float(adf_res[0]),
-            'p_value': float(adf_res[1]),
-            'is_stationary_at_5_percent': adf_res[1] < 0.05,
-            'lags_used': int(adf_res[2])
-        }
-    except Exception as e:
-        report['adf_stationarity_test'] = {'error': str(e)}
-
-    # Finalize the cleaned dataframe
-    cleaned_df = df_ffill.reset_index().rename(columns={'index': TIMESTAMP_COL})
-    
-    print("Audit and cleaning complete.")
-    return cleaned_df, report
-
-def main():
-    """
-    Main execution function.
-    """
-    if not os.path.exists(RAW_CSV):
-        print(f"Error: Raw CSV not found at '{RAW_CSV}'.")
-        print("Please ensure the file is available in the correct path.")
-        return
-
-    setup_directories()
-    raw_df = read_and_prepare_data(RAW_CSV)
-    cleaned_df, audit_report = audit_and_clean(raw_df)
-    
-    # Save the cleaned data
-    cleaned_df.to_csv(CLEAN_CSV, index=False)
-    print(f"Cleaned data written to: {CLEAN_CSV}")
-
-    # Save the audit report
-    with open(REPORT_JSON, 'w') as f:
-        json.dump(audit_report, f, indent=2)
-    print(f"Audit report written to: {REPORT_JSON}")
-    print("\n--- Audit Summary ---")
-    print(json.dumps(audit_report, indent=2))
-    print("---------------------\n")
-
-
-if __name__ == "__main__":
-    main()
+"""
+core/data_ingest.py — Robust ingestion + audit for Phase I
+
+Features & improvements vs. original:
+- CLI (argparse) to specify input/output paths and parameters
+- Flexible timestamp and OHLCV column detection (case-insensitive)
+- Configurable spike detection and rolling window
+- Uses logging module for structured output
+- Atomic write of CSV + JSON report
+- More detailed audit report fields (duplicates, gaps, spike samples)
+
+Example usage:
+  python core/data_ingest.py \
+    --input data/raw/BTCUSDT_1m_December_2024.csv \
+    --out-clean data/clean/BTCUSDT_1m_December_2024.cleaned.csv \
+    --out-report data/reports/BTCUSDT_1m_December_2024_audit.json \
+    --zscore-threshold 6.0 --rolling-window 60
+"""
+from __future__ import annotations
+
+import argparse
+import json
+import logging
+import os
+import sys
+import tempfile
+from typing import Dict, List, Tuple
+
+from datetime import datetime
+import numpy as np
+import pandas as pd
+from statsmodels.tsa.stattools import adfuller
+
+# Default constants (overridden by CLI)
+DEFAULT_FREQ = "1T"
+DEFAULT_TIMEZONE = "UTC"
+DEFAULT_SPIKE_Z = 6.0
+DEFAULT_ROLLING_WINDOW = 60
+DEFAULT_MIN_PERIODS = 10
+
+logger = logging.getLogger("data_ingest")
+logger.setLevel(logging.INFO)
+handler = logging.StreamHandler(sys.stdout)
+handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
+logger.addHandler(handler)
+
+
+def detect_timestamp_column(df: pd.DataFrame) -> str:
+    """Return the best-guess timestamp column name (case-insensitive)."""
+    candidates = [
+        "datetime_utc", "datetime", "timestamp", "time", "date", "date_utc", "datetime_utc"
+    ]
+    cols_lower = {c.lower(): c for c in df.columns}
+    for c in candidates:
+        if c in cols_lower:
+            return cols_lower[c]
+    # Fallback: first datetime-like column
+    for c in df.columns:
+        try:
+            _ = pd.to_datetime(df[c].iloc[0:5])
+            return c
+        except Exception:
+            continue
+    raise ValueError("No timestamp column detected. Please supply --timestamp-col.")
+
+
+def canonicalize_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
+    """
+    Normalize common OHLCV column names to lower-case canonical names:
+    open, high, low, close, volume
+    Returns (df, mapping)
+    """
+    mapping = {}
+    lower_map = {c.lower(): c for c in df.columns}
+    wanted = {"open", "high", "low", "close", "volume"}
+    for w in wanted:
+        if w in lower_map:
+            mapping[lower_map[w]] = w
+        else:
+            # some files use single-letter or capitalized names
+            alt = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
+            for k, v in alt.items():
+                if k in lower_map and v == w:
+                    mapping[lower_map[k]] = v
+    if mapping:
+        df = df.rename(columns=mapping)
+    return df, mapping
+
+
+def setup_directories_for(path: str):
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+
+
+def safe_write_json(path: str, obj):
+    setup_directories_for(path)
+    tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path), mode="w", suffix=".tmp")
+    try:
+        json.dump(obj, tmp, indent=2, default=str)
+        tmp.flush()
+        tmp.close()
+        os.replace(tmp.name, path)
+    finally:
+        if os.path.exists(tmp.name):
+            try:
+                os.remove(tmp.name)
+            except Exception:
+                pass
+
+
+def safe_write_csv(path: str, df: pd.DataFrame, date_col: str = None):
+    setup_directories_for(path)
+    tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path), suffix=".tmp", mode="w")
+    try:
+        if date_col and date_col in df.columns:
+            df.to_csv(tmp.name, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
+        else:
+            df.to_csv(tmp.name, index=False)
+        tmp.flush()
+        tmp.close()
+        os.replace(tmp.name, path)
+    finally:
+        if os.path.exists(tmp.name):
+            try:
+                os.remove(tmp.name)
+            except Exception:
+                pass
+
+
+def read_and_prepare_data(path: str, timestamp_col: str | None = None) -> Tuple[pd.DataFrame, str]:
+    logger.info("Reading raw data from %s", path)
+    df = pd.read_csv(path)
+
+    # detect timestamp column if not provided
+    if not timestamp_col:
+        timestamp_col = detect_timestamp_column(df)
+        logger.info("Auto-detected timestamp column: %s", timestamp_col)
+    else:
+        if timestamp_col not in df.columns:
+            raise ValueError(f"Provided timestamp column '{timestamp_col}' not found in CSV")
+
+    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
+    if df[timestamp_col].isna().any():
+        nbad = int(df[timestamp_col].isna().sum())
+        logger.warning("Found %d unparsable datetime values in column %s", nbad, timestamp_col)
+        df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)
+
+    df = df.sort_values(timestamp_col).reset_index(drop=True)
+    logger.info("Successfully read %d rows.", len(df))
+    return df, timestamp_col
+
+
+def audit_and_clean(
+    df: pd.DataFrame,
+    timestamp_col: str,
+    freq: str = DEFAULT_FREQ,
+    tz: str = DEFAULT_TIMEZONE,
+    spike_z: float = DEFAULT_SPIKE_Z,
+    rolling_window: int = DEFAULT_ROLLING_WINDOW,
+    min_periods: int = DEFAULT_MIN_PERIODS,
+    fill_method: str = "ffill_bfill",
+) -> Tuple[pd.DataFrame, Dict]:
+    logger.info("Starting data audit and cleaning...")
+    report: Dict = {}
+
+    report["rows_raw"] = int(len(df))
+
+    # DEDUPE
+    dup_mask = df.duplicated(subset=[timestamp_col], keep=False)
+    report["duplicate_timestamp_count"] = int(dup_mask.sum())
+    if report["duplicate_timestamp_count"] > 0:
+        logger.info("Removing %d duplicate timestamp rows (keeping last)", report["duplicate_timestamp_count"])
+        df = df.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)
+
+    # canonicalize columns (open/high/low/close/volume)
+    df, mapping = canonicalize_cols(df)
+    report["column_mapping"] = mapping
+    logger.info("Column mapping applied: %s", mapping)
+
+    # set index
+    df = df.set_index(timestamp_col)
+    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=tz)
+    report["first_timestamp"] = str(df.index.min())
+    report["last_timestamp"] = str(df.index.max())
+    report["expected_rows_for_timespan"] = int(len(full_idx))
+    missing_count = int(len(full_idx.difference(df.index)))
+    report["missing_timestamps_count"] = missing_count
+    logger.info("Time span: %s -> %s, expected rows %d, missing %d",
+                report["first_timestamp"], report["last_timestamp"], report["expected_rows_for_timespan"], missing_count)
+
+    # Reindex & mark imputed
+    df_reindexed = df.reindex(full_idx)
+    df_reindexed["_was_missing"] = df_reindexed.isna().any(axis=1)
+
+    # Choose fill strategy
+    fill_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df_reindexed.columns]
+    if fill_method == "ffill_bfill":
+        df_filled = df_reindexed.copy()
+        if fill_cols:
+            df_filled[fill_cols] = df_filled[fill_cols].ffill().bfill()
+    elif fill_method == "interpolate":
+        df_filled = df_reindexed.copy()
+        if fill_cols:
+            df_filled[fill_cols] = df_filled[fill_cols].interpolate(limit_direction="both")
+    else:
+        df_filled = df_reindexed.copy()
+
+    imputed_mask = df_filled[fill_cols].isna().any(axis=1) == False
+    # is_imputed_datapoint marks rows that were originally missing and were filled
+    df_filled["is_imputed_datapoint"] = df_reindexed["_was_missing"].astype(int)
+    logger.info("Performed fill on missing timestamps using method '%s'.", fill_method)
+
+    # Spike detection using rolling z-score on 'close' if available
+    if "close" in df_filled.columns:
+        roll_mean = df_filled["close"].rolling(window=rolling_window, min_periods=min_periods).mean()
+        roll_std = df_filled["close"].rolling(window=rolling_window, min_periods=min_periods).std()
+        # avoid divide-by-zero
+        eps = 1e-12
+        roll_std = roll_std.fillna(0).replace(0, eps)
+        zscore = (df_filled["close"] - roll_mean) / roll_std
+        df_filled["spike_zscore"] = zscore
+        spike_mask = zscore.abs() > spike_z
+        df_filled["spike_candidate"] = spike_mask.fillna(False)
+        report["spike_candidates_found"] = int(spike_mask.sum())
+        logger.info("Identified %d spike candidates (z>%s).", report["spike_candidates_found"], spike_z)
+        # attach a few example spike rows to the report
+        if report["spike_candidates_found"] > 0:
+            sample_spikes = df_filled.loc[spike_mask].head(10)
+            report["spike_examples"] = [
+                {
+                    "timestamp": str(idx),
+                    "close": float(row["close"]),
+                    "zscore": float(row["spike_zscore"])
+                }
+                for idx, row in sample_spikes[["close", "spike_zscore"]].iterrows()
+            ]
+    else:
+        logger.warning("No 'close' column found; skipping spike detection.")
+        report["spike_candidates_found"] = 0
+
+    # log-return calculation
+    if "close" in df_filled.columns:
+        df_filled["log_return_1m"] = np.log(df_filled["close"] / df_filled["close"].shift(1))
+        df_filled["log_return_1m"] = df_filled["log_return_1m"].replace([np.inf, -np.inf], np.nan).fillna(0)
+        lr = df_filled["log_return_1m"].replace([np.inf, -np.inf], np.nan).dropna()
+        report["logreturn_mean"] = float(lr.mean()) if len(lr) > 0 else None
+        report["logreturn_std"] = float(lr.std()) if len(lr) > 0 else None
+        report["logreturn_skew"] = float(lr.skew()) if len(lr) > 0 else None
+        report["logreturn_kurtosis"] = float(lr.kurtosis()) if len(lr) > 0 else None
+    else:
+        report["logreturn_mean"] = report["logreturn_std"] = report["logreturn_skew"] = report["logreturn_kurtosis"] = None
+
+    # ADF test where appropriate
+    try:
+        if "log_return_1m" in df_filled.columns and len(df_filled["log_return_1m"].dropna()) > 50:
+            adf_res = adfuller(df_filled["log_return_1m"].dropna(), autolag="AIC")
+            report["adf_stationarity_test"] = {
+                "statistic": float(adf_res[0]),
+                "p_value": float(adf_res[1]),
+                "is_stationary_at_5_percent": adf_res[1] < 0.05,
+                "lags_used": int(adf_res[2]),
+            }
+        else:
+            report["adf_stationarity_test"] = {"error": "insufficient data for ADF"}
+    except Exception as e:
+        report["adf_stationarity_test"] = {"error": str(e)}
+
+    cleaned_df = df_filled.reset_index().rename(columns={"index": timestamp_col})
+    logger.info("Audit and cleaning complete.")
+    return cleaned_df, report
+
+
+def parse_args():
+    p = argparse.ArgumentParser(description="Data ingestion & audit (Phase I)")
+    p.add_argument("--input", "-i", required=True, help="Path to raw CSV (e.g. data/raw/filename.csv)")
+    p.add_argument("--out-clean", "-c", required=True, help="Path to write cleaned CSV (e.g. data/clean/...)")
+    p.add_argument("--out-report", "-r", required=True, help="Path to write JSON audit report")
+    p.add_argument("--timestamp-col", help="Name of timestamp column (optional)", default=None)
+    p.add_argument("--freq", help="Pandas frequency for reindexing", default=DEFAULT_FREQ)
+    p.add_argument("--tz", help="Timezone name for index (default UTC)", default=DEFAULT_TIMEZONE)
+    p.add_argument("--zscore-threshold", type=float, default=DEFAULT_SPIKE_Z, help="Z-score threshold for spikes")
+    p.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="Rolling window (minutes)")
+    p.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS, help="min periods for rolling stats")
+    p.add_argument("--fill-method", choices=["ffill_bfill", "interpolate", "none"], default="ffill_bfill")
+    return p.parse_args()
+
+
+def main():
+    args = parse_args()
+    try:
+        df, timestamp_col = read_and_prepare_data(args.input, timestamp_col=args.timestamp_col)
+    except Exception as e:
+        logger.error("Failed to read input CSV: %s", e)
+        sys.exit(2)
+
+    cleaned_df, report = audit_and_clean(
+        df=df,
+        timestamp_col=timestamp_col,
+        freq=args.freq,
+        tz=args.tz,
+        spike_z=args.zscore_threshold,
+        rolling_window=args.rolling_window,
+        min_periods=args.min_periods,
+        fill_method=args.fill_method,
+    )
+
+    # Persist outputs atomically
+    try:
+        safe_write_csv(args.out_clean, cleaned_df, date_col=timestamp_col)
+        safe_write_json(args.out_report, report)
+        logger.info("Wrote cleaned CSV to: %s", args.out_clean)
+        logger.info("Wrote audit report to: %s", args.out_report)
+        logger.info("Summary: %s", json.dumps(report, indent=2)[:1000])
+    except Exception as e:
+        logger.exception("Failed to write outputs: %s", e)
+        sys.exit(3)
+
+
+if __name__ == "__main__":
+    main()
-- 
2.34.1
