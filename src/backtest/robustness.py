from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from .backtester import VectorizedBacktester, load_cfg, load_cleaned, load_signals


@dataclass
class RobustnessResult:
    name: str
    payload: dict

def save_result(outdir: Path, name: str, payload: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(payload, indent=2))

def run_oos(df: pd.DataFrame, cfg: Dict, outdir: Path) -> RobustnessResult:
    frac = float(cfg["oos"]["test_fraction"])
    split_idx = int(len(df) * (1 - frac))
    df_oos = df.iloc[split_idx:].copy()
    signals = load_signals(cfg_parent, df_oos)  # reuse model/heuristic
    b = VectorizedBacktester(cfg_parent, outputs_dir=str(outdir / "oos"))
    res = b.run(df_oos, signals)
    save_result(outdir, "oos_summary", res["summary"])
    return RobustnessResult("oos", res["summary"])

def run_walk_forward(df: pd.DataFrame, cfg: Dict, outdir: Path) -> RobustnessResult:
    td = int(cfg["walk_forward"]["train_days"]) * 24 * 60
    sd = int(cfg["walk_forward"]["test_days"]) * 24 * 60
    step = int(cfg["walk_forward"]["step_days"]) * 24 * 60
    retrain = bool(cfg["walk_forward"]["retrain"])

    equity_chunks = []
    summaries = []
    i = 0
    while i + td + sd <= len(df):
        df_train = df.iloc[i : i + td]
        df_test = df.iloc[i + td : i + td + sd]
        # NOTE: retrain hook is left as placeholder; Phase III artifacts would be loaded & optionally retrained here.
        signals = load_signals(cfg_parent, df_test)
        b = VectorizedBacktester(cfg_parent, outputs_dir=str(outdir / f"wf_{i}"))
        res = b.run(df_test, signals)
        equity_chunks.append(res["equity_curve"]["equity"])
        summaries.append(res["summary"])
        i += step
    if not equity_chunks:
        return RobustnessResult("walk_forward", {"error": "insufficient data"})
    equity = pd.concat(equity_chunks).reset_index(drop=True)
    summary = {
        "segments": len(summaries),
        "median_final_equity": float(np.median([s["final_equity"] for s in summaries])),
        "median_sharpe": float(np.median([s["sharpe"] for s in summaries])),
        "median_profit_factor": float(np.median([s["profit_factor"] for s in summaries])),
    }
    save_result(outdir, "walk_forward_summary", summary)
    return RobustnessResult("walk_forward", summary)

def run_monte_carlo(trades: pd.Series, cfg: Dict, outdir: Path) -> RobustnessResult:
    n = int(cfg["monte_carlo"]["reshuffles"])
    seed = int(cfg["monte_carlo"]["seed"])
    rng = np.random.default_rng(seed)
    pnl = trades.dropna().to_numpy()
    if pnl.size == 0:
        return RobustnessResult("monte_carlo", {"error": "no trades"})
    totals = []
    for _ in range(n):
        reshuffled = rng.permutation(pnl)
        totals.append(float(reshuffled.sum()))
    lo, med, hi = np.percentile(totals, [2.5, 50, 97.5])
    out = {
        "reshuffles": n,
        "total_pnl_ci_95": [float(lo), float(hi)],
        "median_total_pnl": float(med),
        "negative_median": bool(med < 0)
    }
    save_result(outdir, "monte_carlo_summary", out)
    return RobustnessResult("monte_carlo", out)

def run_sensitivity(base_cfg: Dict, df: pd.DataFrame, cfg: Dict, outdir: Path) -> RobustnessResult:
    sweep = float(cfg["sensitivity"]["sweep_percent"])
    params = cfg["sensitivity"]["params"]
    records = []
    for p in params:
        name = p["name"]
        path = p["path"].split(".")
        # get original
        orig = base_cfg
        for k in path:
            orig = orig[k]
        for sgn in [-sweep, 0.0, sweep]:
            # clone
            mod = json.loads(json.dumps(base_cfg))
            ref = mod
            for k in path[:-1]:
                ref = ref[k]
            ref[path[-1]] = (1 + sgn) * orig if isinstance(orig, (int, float)) else orig
            # run
            signals = load_signals(mod, df)
            b = VectorizedBacktester(mod, outputs_dir=str(outdir / f"sens_{name}_{sgn:+.2f}"))
            res = b.run(df, signals)
            rec = {"param": name, "delta": float(sgn), **res["summary"]}
            records.append(rec)
    df_out = pd.DataFrame(records)
    df_out.to_csv(outdir / "sensitivity.csv", index=False)
    # cliff detection heuristic: any metric drops >50% when +/-20% param tweak
    flag = False
    for name in set(df_out["param"]):
        base = df_out[(df_out["param"]==name) & (df_out["delta"]==0.0)]
        if base.empty:
            continue
        b_sh = float(base["sharpe"].iloc[0])
        worst = df_out[df_out["param"]==name]["sharpe"].min()
        if b_sh > 0 and (worst < 0.5 * b_sh):
            flag = True
    out = {"cliff_like_sensitivity": bool(flag)}
    save_result(outdir, "sensitivity_summary", out)
    return RobustnessResult("sensitivity", out)

def run_noise(df: pd.DataFrame, cfg: Dict, outdir: Path) -> RobustnessResult:
    runs = int(cfg["noise"]["runs"])
    mult = float(cfg["noise"]["std_mult"])
    ret = df["close"].pct_change().dropna()
    std = ret.std()
    eq_ends = []
    for i in range(runs):
        noisy = df.copy()
        noise = np.random.normal(0.0, mult * std, size=len(noisy))
        noisy["close"] = noisy["close"] * (1 + pd.Series(noise, index=noisy.index)).clip(lower=1e-9)
        # maintain OHLC coherence (simple)
        mid = noisy["close"]
        rng = noisy["high"] - noisy["low"]
        noisy["open"] = mid.shift(1).fillna(mid)
        noisy["high"] = np.maximum(mid + 0.25*rng, mid)
        noisy["low"]  = np.minimum(mid - 0.25*rng, mid)
        signals = load_signals(cfg_parent, noisy)
        b = VectorizedBacktester(cfg_parent, outputs_dir=str(outdir / f"noise_{i}"))
        res = b.run(noisy, signals)
        eq_ends.append(res["summary"]["final_equity"])
    out = {
        "runs": runs,
        "median_final_equity": float(np.median(eq_ends)),
        "p05_final_equity": float(np.percentile(eq_ends, 5)),
        "p95_final_equity": float(np.percentile(eq_ends, 95))
    }
    save_result(outdir, "noise_summary", out)
    return RobustnessResult("noise", out)

# Global to reuse model/signal loader semantics
cfg_parent: Dict = {}

def run_all(config_path: str, robustness_cfg_path: str, cleaned_csv: str | None = None, outdir: str = "artifacts/robustness/") -> List[RobustnessResult]:
    global cfg_parent
    cfg_parent = load_cfg(config_path)
    if cleaned_csv:
        cfg_parent["paths"]["cleaned_csv"] = cleaned_csv
    df = load_cleaned(cfg_parent["paths"]["cleaned_csv"])
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    robocfg = load_cfg(robustness_cfg_path)

    # Baseline backtest to collect trades
    signals = load_signals(cfg_parent, df)
    b = VectorizedBacktester(cfg_parent, outputs_dir=str(outdir / "baseline"))
    baseline = b.run(df, signals)
    trades = baseline["trades"]["trade_pnl"]

    results = []
    results.append(run_oos(df, robocfg, outdir))
    results.append(run_walk_forward(df, robocfg, outdir))
    results.append(run_monte_carlo(trades, robocfg, outdir))
    results.append(run_sensitivity(cfg_parent, df, robocfg, outdir))
    results.append(run_noise(df, robocfg, outdir))

    # diagnostics gate
    gate = {
        "baseline_sharpe": baseline["summary"]["sharpe"],
        "baseline_profit_factor": baseline["summary"]["profit_factor"],
        "monte_carlo_median_positive": results[2].payload.get("negative_median") is False
    }
    (Path(outdir) / "gate_summary.json").write_text(json.dumps(gate, indent=2))
    return results
