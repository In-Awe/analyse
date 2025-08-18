#!/usr/bin/env python3
"""
Create classification/regression targets from cleaned OHLCV CSV.

Produces:
 - features + target parquet for downstream training:
   data/features_with_target.parquet

Target: next N-minute log return direction:
  UP (1) if log_ret_next > tau
  DOWN (-1) if log_ret_next < -tau
  SIDEWAYS (0) otherwise

Config: configs/target.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

EPS = 1e-12

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def create_target(df: pd.DataFrame, horizon: int, tau: float) -> pd.DataFrame:
    # compute log returns
    logp = np.log(df["close"])
    future_logp = logp.shift(-horizon)
    logret = (future_logp - logp).fillna(0.0)
    # create numeric target
    target = np.where(logret > tau, 1, np.where(logret < -tau, -1, 0))
    df = df.copy()
    df[f"next_{horizon}m_logret"] = logret
    df[f"next_{horizon}m_dir"] = target
    return df

def time_splits(df: pd.DataFrame, cfg: dict):
    n = len(df)
    train_end = int(n * cfg["splits"]["train_frac"])
    val_end = train_end + int(n * cfg["splits"]["val_frac"])
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="cleaned CSV input")
    p.add_argument("--config", default="configs/target.yaml", help="target config yaml")
    p.add_argument("--out", default="data/features/features_with_target.parquet", help="output parquet")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    horizon = int(cfg["target"]["horizon_minutes"])
    tau = float(cfg["target"]["tau"])
    df = read_clean_csv(args.input)
    df_t = create_target(df, horizon, tau)
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df_t.to_parquet(outpath, index=False)
    splits = time_splits(df_t, cfg["target"])
    meta = {"rows": len(df_t), "horizon": horizon, "tau": tau, "splits": splits}
    print(f"[target] wrote {outpath} rows={len(df_t)} meta={meta}")

if __name__ == "__main__":
    main()
