#!/usr/bin/env python3
"""
Convenience CLI to run Phase II feature build:
  - reads cleaned CSV
  - builds features per configs/features.yaml
  - writes parquet to data/features/
Usage:
  python scripts/build_features.py --input data/cleaned/BTCUSD_1min.cleaned.csv --out data/features/technical.parquet
"""
import argparse
from pathlib import Path
import json
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from src.features.engine import build_features, read_clean_csv, load_config

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="cleaned CSV (from Phase I)")
    p.add_argument("--config", default="configs/features.yaml", help="features config yaml")
    p.add_argument("--out", default="data/features/technical.parquet", help="output parquet")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    print(f"[build_features] loading {args.input}")
    df = read_clean_csv(args.input)
    print(f"[build_features] building features using {args.config}")
    feats = build_features(df, cfg)
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(outpath, index=False)
    print(f"[build_features] written {len(feats)} rows to {outpath}")

if __name__ == "__main__":
    main()
