#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from src.backtest.backtester import VectorizedBacktester, load_cfg, load_cleaned, load_signals

console = Console()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trade_logic.yaml")
    ap.add_argument("--cleaned", default=None, help="override cleaned csv path")
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.cleaned:
        cfg["paths"]["cleaned_csv"] = args.cleaned
    df = load_cleaned(cfg["paths"]["cleaned_csv"])
    signals = load_signals(cfg, df)

    b = VectorizedBacktester(cfg, outputs_dir=cfg["paths"]["outputs_dir"])
    res = b.run(df, signals, initial_equity=args.initial_equity)

    console.print(Panel.fit(f"[bold]Backtest complete[/bold]\n"
                            f"Sharpe: {res['summary']['sharpe']:.3f}\n"
                            f"Sortino: {res['summary']['sortino']:.3f}\n"
                            f"Calmar: {res['summary']['calmar']:.3f}\n"
                            f"MaxDD: {res['summary']['max_drawdown']:.3f}\n"
                            f"PF: {res['summary']['profit_factor']:.3f}\n"
                            f"WinRate: {res['summary']['win_rate']:.3f}\n"
                            f"Trades: {res['summary']['total_trades']}"))

if __name__ == "__main__":
    main()
