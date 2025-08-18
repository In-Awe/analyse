#!/usr/bin/env python3
"""
Runner script to execute Phase III flow:
  1) create target (calls src/models/target.py)
  2) build features with target present (expects data/features/technical.parquet to exist)
  3) train LR, XGBoost, LSTM and save artifacts.

This is a convenience script for Jules to run the full Phase III pipeline.
"""
import argparse
import subprocess
from pathlib import Path

def run(cmd):
    print(f"[run] {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned", default="data/cleaned/BTCUSD_1min.cleaned.csv")
    p.add_argument("--features", default="data/features/technical.parquet")
    p.add_argument("--features_with_target", default="data/features/features_with_target.parquet")
    args = p.parse_args()
    # 1) create target parquet
    run(f"python src/models/target.py --input {args.cleaned} --out {args.features_with_target}")
    # 2) make sure features parquet exists (assumes Phase II executed); if not, try to build
    if not Path(args.features).exists():
        print("[train_models] features parquet missing, attempting to run Phase II build_features")
        run(f"python scripts/build_features.py --input {args.cleaned} --out {args.features}")
    # 3) merge features and target: naive join on timestamp
    import pandas as pd
    feats = pd.read_parquet(args.features)
    targ = pd.read_parquet(args.features_with_target)
    merged = targ.merge(feats, on="timestamp", how="left")
    outpath = Path(args.features_with_target).parent / "merged_features_with_target.parquet"
    merged.to_parquet(outpath, index=False)
    print(f"[train_models] wrote merged features to {outpath}")
    # 4) train models
    run(f"python src/models/train_lr.py --features {outpath} --out-dir artifacts/models/lr")
    run(f"python src/models/train_xgb.py --features {outpath} --out-dir artifacts/models/xgb")
    run(f"python src/models/train_lstm.py --features {outpath} --out-dir artifacts/models/lstm --epochs 5")
    print("[train_models] done")

if __name__ == "__main__":
    main()
