#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from src.backtest.robustness import run_all

console = Console()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trade_logic.yaml")
    ap.add_argument("--robustness", default="configs/robustness.yaml")
    ap.add_argument("--cleaned", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    results = run_all(args.config, args.robustness, cleaned_csv=args.cleaned)
    tbl = Table(title="Robustness Summary")
    tbl.add_column("Test")
    tbl.add_column("Key Output")
    for r in results:
        key = ", ".join([f"{k}={v}" for k, v in list(r.payload.items())[:3]])
        tbl.add_row(r.name, key)
    console.print(tbl)

if __name__ == "__main__":
    main()
