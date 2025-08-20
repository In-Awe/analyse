#!/usr/bin/env bash
# scripts/train_on_downloads.py
# Simulate training across downloaded artifacts. Reads artifacts/raw/<SYMBOL>/*.csv,
# builds a simple equity_curve and trades.csv per-symbol, writes artifacts/models/<symbol>_stub.dill,
# and uses tools/write_summary_atomic.py to produce summary.json for each symbol.
#
# Usage:
#  python scripts/train_on_downloads.py --raw artifacts/raw --limit 50
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT"
export PYTHONPATH="${PY}:${PYTHONPATH:-}"

python - <<'PY'
import argparse, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import dill
import time
from tools.write_summary_atomic import write_summary_for_equity

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="artifacts/raw")
parser.add_argument("--out_models", default="artifacts/models")
parser.add_argument("--out_train", default="artifacts/training")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

raw = Path(args.raw)
symbols = [p.name for p in raw.iterdir() if p.is_dir()]
symbols.sort()
if args.limit>0:
    symbols = symbols[:args.limit]

print("Found symbols to train on:", symbols)

for sym in symbols:
    sym_dir = raw / sym
    # find all monthly CSVs and concatenate
    csvs = sorted(sym_dir.glob(f"{sym}_1m_*.csv"))
    if not csvs:
        print("No CSVs for", sym, "skipping")
        continue
    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c, parse_dates=['open_time'])
            dfs.append(df)
        except Exception as e:
            print("Failed reading", c, e)
    if not dfs:
        continue
    df_all = pd.concat(dfs, ignore_index=True)
    # build a toy equity curve: start 10000, apply cumulative returns of pct_change(close)*0.1 to simulate trading
    df_all['close'] = df_all['close'].astype(float)
    df_all['pct'] = df_all['close'].pct_change().fillna(0)
    # simulate incremental returns
    df_all['sim_ret'] = df_all['pct'] * 0.1  # scale down
    equity = 10000 * (1 + df_all['sim_ret']).cumprod()
    eq_df = pd.DataFrame({"timestamp": df_all['open_time'], "equity": equity})
    # save into artifacts/training/<sym>/equity_curve.csv
    out_dir = Path(args.out_train) / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_path = out_dir / "equity_curve.csv"
    eq_df.to_csv(eq_path, index=False)
    # create simple trades file from sim_ret peaks
    trades = df_all[df_all['sim_ret'] != 0].copy()
    trades['pnl'] = (trades['sim_ret'] * 100).round(6)  # small pnl proxy
    trades_path = out_dir / "trades.csv"
    trades[['open_time','pnl']].rename(columns={'open_time':'timestamp'}).to_csv(trades_path, index=False)
    # write a tiny model artifact
    model_dir = Path(args.out_models)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{sym}_stub.dill"
    model_obj = {"symbol": sym, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": 42}
    with open(model_path, "wb") as f:
        dill.dump(model_obj, f)
    print("Wrote model for", sym, "and equity:", eq_path)
    # call atomic writer to create summary.json next to eq_path
    try:
        write_summary_for_equity(eq_path, trades_path)
        print("Wrote summary.json for", sym)
    except Exception as e:
        print("Failed to write summary for", sym, e)

print("Training-simulation complete.")
PY
