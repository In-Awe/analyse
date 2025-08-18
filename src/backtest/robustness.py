#!/usr/bin/env python3
"""
Robustness tools for Phase IV:
 - Monte Carlo trade reshuffling: randomize trade order (preserve P&L magnitudes) and compute distribution of summary stats
   Supports: single-threaded, multiprocessing, or Dask distributed execution.
 - Parameter sensitivity: sweep specified parameters and compute resulting backtest summaries. Parallelized.
"""
from __future__ import annotations
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import copy

import numpy as np
import pandas as pd

from src.backtest.backtester import vectorized_backtest, load_merged_features, load_config
import multiprocessing as mp
from tqdm.auto import tqdm

try:
    import dask
    from dask import delayed, compute
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except Exception:
    DASK_AVAILABLE = False

def _mc_reshuffle_worker(args: Tuple[int, int, np.ndarray, int]) -> Dict[str, float]:
    """Helper for multiprocessing, takes (i, seed, pl)"""
    i, seed, pl = args
    rnd = np.random.default_rng(seed + i)
    perm = rnd.permutation(pl)
    cum = np.cumsum(perm)
    total = float(cum[-1])
    peak = float(np.max(cum))
    trough = float(np.min(cum))
    maxdd = float((trough - peak) if trough < peak else 0.0)
    return {"total_pnl": total, "max_dd": maxdd}

def monte_carlo_trade_reshuffle(trades_df: pd.DataFrame, n: int = 1000, seed: int = 42, mode: str = "auto", n_workers: Optional[int] = None, dask_cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Monte Carlo reshuffle of trades P&L.
    """
    pl = trades_df["pnl"].values if len(trades_df) > 0 else np.array([])
    if pl.size == 0:
        return {"n": n, "note": "no trades to reshuffle", "results": []}

    mode_eff = mode
    if mode == "auto":
        if dask_cfg and dask_cfg.get("use_dask", False) and DASK_AVAILABLE:
            mode_eff = "dask"
        else:
            mode_eff = "multiprocess"

    def single_run(i, seed_local):
        rnd = np.random.default_rng(seed_local + i)
        perm = rnd.permutation(pl)
        cum = np.cumsum(perm)
        total = float(cum[-1])
        peak = float(np.max(cum))
        trough = float(np.min(cum))
        maxdd = float((trough - peak) if trough < peak else 0.0)
        return {"total_pnl": total, "max_dd": maxdd}

    results = []
    if mode_eff == "single":
        for i in tqdm(range(n)):
            results.append(single_run(i, seed))
    elif mode_eff == "multiprocess":
        workers = n_workers or max(1, mp.cpu_count() - 1)
        with mp.Pool(processes=workers) as pool:
            args = [(i, seed, pl) for i in range(n)]
            results = list(tqdm(pool.imap_unordered(_mc_reshuffle_worker, args), total=n))
    elif mode_eff == "dask":
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask requested but dask/distributed not installed.")
        client = None
        if dask_cfg and dask_cfg.get("address"):
            client = Client(address=dask_cfg["address"])
        else:
            workers = n_workers or max(1, mp.cpu_count() - 1)
            cluster = LocalCluster(n_workers=workers, threads_per_worker=1, processes=True)
            client = Client(cluster)

        tasks = [delayed(single_run)(i, seed) for i in range(n)]
        results = list(compute(*tasks, scheduler="distributed")[0])
        client.close()
    else:
        raise ValueError(f"unknown mode: {mode_eff}")

    totals = np.array([r["total_pnl"] for r in results])
    dd = np.array([r["max_dd"] for r in results])
    out = {
        "n": n,
        "mode": mode_eff,
        "total_pnl_mean": float(totals.mean()),
        "total_pnl_median": float(np.median(totals)),
        "total_pnl_p2_5": float(np.percentile(totals, 2.5)),
        "total_pnl_p97_5": float(np.percentile(totals, 97.5)),
        "maxdd_median": float(np.median(dd)),
    }
    return {"summary": out, "raw_sample": results[:min(1000, len(results))]}

def _ps_grid_worker(args: Tuple[str, str, Dict[str, Any]]):
    """Helper for multiprocessing parameter sensitivity"""
    cfg_path, merged_path, override = args
    cfg_local = load_config(cfg_path)
    c = copy.deepcopy(cfg_local)
    for k, v in override.items():
        parts = k.split(".")
        node = c
        for p in parts[:-1]:
            if p not in node:
                node[p] = {}
            node = node[p]
        node[parts[-1]] = v
    flat = c.get("backtest", {})
    df_local = load_merged_features(merged_path)
    res = vectorized_backtest(df_local, flat)
    return {"params": override, "summary": res["summary"]}

def parameter_sensitivity_grid(cfg_path: str, merged_path: str, param_grid: List[Dict[str, Any]], parallel: bool=False, n_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    if not parallel:
        results = []
        for pg in tqdm(param_grid):
            results.append(_ps_grid_worker((cfg_path, merged_path, pg)))
        return results
    else:
        workers = n_workers or max(1, mp.cpu_count() - 1)
        with mp.Pool(processes=workers) as pool:
            args_list = [(cfg_path, merged_path, pg) for pg in param_grid]
            results = list(tqdm(pool.imap_unordered(_ps_grid_worker, args_list), total=len(param_grid)))
        return results
