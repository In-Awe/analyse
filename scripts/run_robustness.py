#!/usr/bin/env python3
"""
Runner for robustness analysis:
 - Run backtester to obtain baseline trades
 - Run Monte Carlo trade reshuffling
 - Run parameter sensitivity grid
Outputs saved under artifacts/backtest/robustness_<timestamp>/
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtest.backtester import vectorized_backtest, load_config, load_merged_features
from src.backtest.robustness import monte_carlo_trade_reshuffle, parameter_sensitivity_grid

def run(args):
    cfg = load_config(args.config)
    merged = args.merged or cfg["backtest"]["input"]["merged_features"]
    out_root = Path(cfg["backtest"]["outputs"]["out_dir"])
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"robustness_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_merged_features(merged)
    flat_cfg = cfg["backtest"]
    # run baseline backtest
    print("[robustness] running baseline backtest")
    res = vectorized_backtest(df, flat_cfg)
    trades = res["trades"]
    # save baseline
    trades.to_csv(out_dir / "baseline_trades.csv", index=False)
    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump(res["summary"], f, indent=2)
    # Monte Carlo
    mc_cfg = cfg.get("robustness", {}).get("monte_carlo", {})
    n = int(mc_cfg.get("n_shuffles", 2000))
    seed = int(mc_cfg.get("seed", 42))
    print(f"[robustness] running monte-carlo reshuffle n={n}")
    mc = monte_carlo_trade_reshuffle(trades, n=n, seed=seed)
    with open(out_dir / "monte_carlo.json", "w") as f:
        json.dump(mc, f, indent=2)
    # parameter sensitivity
    ps_cfg = cfg.get("robustness", {}).get("param_sensitivity", {})
    sweep = ps_cfg.get("sweep", [])
    # build grid of parameter dicts
    from itertools import product
    names = [s["name"] for s in sweep]
    lists = [s["values"] for s in sweep]
    grid = []
    for combo in product(*lists):
        d = {names[i]: combo[i] for i in range(len(names))}
        grid.append(d)
    print(f"[robustness] running param sensitivity for {len(grid)} combinations")
    ps_results = parameter_sensitivity_grid(args.config, merged, grid)
    with open(out_dir / "param_sensitivity.json", "w") as f:
        json.dump(ps_results, f, indent=2)
    print(f"[robustness] saved results to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--merged", default=None, help="override merged parquet path")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
