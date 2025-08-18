#!/usr/bin/env python3
"""
Convenience runner for Phase IV backtester scaffold.

Usage:
  python scripts/run_backtest.py --config configs/backtest.yaml
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
    p.add_argument("--config", default="configs/backtest.yaml")
    p.add_argument("--merged", default=None, help="override merged parquet path")
    args = p.parse_args()
    cmd = f"python -m src.backtest.backtester --config {args.config}"
    if args.merged:
        cmd += f" --merged {args.merged}"
    run(cmd)
    print("[run_backtest] done. check artifacts/backtest/ for outputs.")
    print("[run_backtest] note: to run robustness analyses use scripts/run_robustness.py")

if __name__ == "__main__":
    main()
