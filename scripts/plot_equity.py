#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-csv", default="artifacts/backtest/equity_curve.csv")
    ap.add_argument("--out", default="artifacts/backtest/equity_curve.png")
    args = ap.parse_args()

    df = pd.read_csv(args.equity_csv, parse_dates=["timestamp"]) if "timestamp" in open(args.equity_csv).readline() else pd.read_csv(args.equity_csv)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    plt.figure()
    df["equity"].plot()
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=144)

if __name__ == "__main__":
    main()
