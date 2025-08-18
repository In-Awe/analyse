#!/usr/bin/env python3
"""
Robustness tools for Phase IV:
 - Monte Carlo trade reshuffling: randomize trade order (preserve P&L magnitudes) and compute distribution of summary stats
 - Parameter sensitivity: sweep specified parameters and compute resulting summary stats via backtester

Usage:
  from src.backtest.robustness import monte_carlo_trade_reshuffle, parameter_sensitivity_grid
  or use scripts/run_robustness.py CLI wrapper.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from src.backtest.backtester import vectorized_backtest, load_merged_features, load_config

def monte_carlo_trade_reshuffle(trades_df: pd.DataFrame, n: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Given a trades DataFrame with 'pnl' column, reshuffle the order of trade P&L n times.
    For each reshuffle compute total P&L and simple metrics distribution.
    Returns summary dict with distribution arrays and percentiles.
    """
    rng = np.random.default_rng(seed)
    pl = trades_df["pnl"].values if len(trades_df) > 0 else np.array([])
    if pl.size == 0:
        return {"n": n, "note": "no trades to reshuffle", "results": []}
    results = []
    for i in range(n):
        perm = rng.permutation(pl)
        cum = np.cumsum(perm)
        total = cum[-1]
        peak = np.max(cum)
        trough = np.min(cum)
        maxdd = trough - peak if trough < peak else 0.0
        results.append({"total_pnl": float(total), "max_dd": float(maxdd)})
    totals = np.array([r["total_pnl"] for r in results])
    dd = np.array([r["max_dd"] for r in results])
    out = {
        "n": n,
        "total_pnl_mean": float(totals.mean()),
        "total_pnl_median": float(np.median(totals)),
        "total_pnl_p2_5": float(np.percentile(totals, 2.5)),
        "total_pnl_p97_5": float(np.percentile(totals, 97.5)),
        "maxdd_median": float(np.median(dd)),
    }
    return {"summary": out, "raw": results}

def parameter_sensitivity_grid(cfg_path: str, merged_path: str, param_grid: List[Dict[str, Any]], parallel: bool=False) -> List[Dict[str, Any]]:
    """
    Run vectorized_backtest across a grid of parameter modifications.
    param_grid: list of dicts where each dict contains keys to override in the backtest config (nested keys dot-separated allowed).
    Returns list of results: each entry contains 'params' and 'summary' (backtest summary).
    """
    cfg = load_config(cfg_path)
    df = load_merged_features(merged_path)
    results = []
    for pg in param_grid:
        # deep copy cfg and apply overrides
        import copy
        c = copy.deepcopy(cfg)
        # pg keys support dot-notation e.g., 'capital.position_fraction'
        for k, v in pg.items():
            parts = k.split(".")
            node = c
            for p in parts[:-1]:
                if p not in node:
                    node[p] = {}
                node = node[p]
            node[parts[-1]] = v
        # flatten to backtest dict structure expected by vectorized_backtest (function expects flattened dict)
        # reuse backtester flattening by writing temp config file and calling vectorized_backtest via main loader
        # to avoid code duplication, call vectorized_backtest directly with flattened c["backtest"]
        flat = c.get("backtest", {})
        res = vectorized_backtest(df, flat)
        results.append({"params": pg, "summary": res["summary"]})
    return results
