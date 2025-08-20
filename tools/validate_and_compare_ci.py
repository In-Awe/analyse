#!/usr/bin/env python3
"""
CI helper: search the entire artifacts/ directory for summary.json files
(not just artifacts/batch), recompute metrics from equity_curve.csv, and
compare to summary.json values. Exit non-zero if:
  - no summary.json files discovered, or
  - any metric mismatches exceed tolerances.

This makes CI robust whether your backtests write into artifacts/batch/*,
artifacts/backtest/run*/ or any other subdir.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict, List

# Reuse the project metric calculator if available.
# It should provide: compute_metrics_from_equity(equity_csv: str, trades_csv: Optional[str]) -> Dict[str, float]
try:
    from tools.validate_metrics import compute_metrics_from_equity
except Exception as e:
    print("ERROR: tools.validate_metrics not importable. Ensure it exists and is installable.", file=sys.stderr)
    raise

# Absolute tolerances for numeric comparisons
TOL = {
    'sharpe': 0.2,
    'sortino': 0.5,
    'max_drawdown': 0.05,
    'profit_factor': 0.2,
    'final_equity': 1e-6,
}

def within_tolerance(refv, compv, key) -> bool:
    try:
        reff = float(refv)
        compf = float(compv)
    except Exception:
        # fallback to exact string equality for non-numeric
        return str(refv) == str(compv)
    tol = TOL.get(key)
    if tol is None:
        # default small relative tolerance
        return abs(reff - compf) <= max(1e-6, 0.01 * abs(reff))
    return abs(reff - compf) <= tol

def compare_dicts(ref: Dict, recomputed: Dict) -> List[str]:
    keys = ('sharpe','sortino','max_drawdown','profit_factor','final_equity')
    mismatches = []
    for k in keys:
        refv = ref.get(k, None)
        compv = recomputed.get(k, None)
        if refv is None:
            mismatches.append(f"Reference missing {k}")
            continue
        if compv is None:
            mismatches.append(f"Recomputed missing {k}")
            continue
        if not within_tolerance(refv, compv, k):
            mismatches.append(f"{k} mismatch: ref={refv} vs recalc={compv} (tol={TOL.get(k,'rel1%')})")
    return mismatches

def main() -> int:
    root = Path("artifacts")
    if not root.exists():
        print("No artifacts/ directory found in repo root. Failing CI.")
        return 2

    summaries = list(root.rglob("summary.json"))
    if not summaries:
        print("ERROR: No summary.json files found anywhere under artifacts/.")
        print("Searched path:", str(root.resolve()))
        return 2

    total_failures = 0
    for s in sorted(summaries):
        print(f"[validate] Checking {s}")
        try:
            data = json.loads(s.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  Could not read/parse: {e}")
            total_failures += 1
            continue
        parent = s.parent
        equity = parent / "equity_curve.csv"
        trades = parent / "trades.csv"
        if not equity.exists():
            print(f"  Missing equity_curve.csv next to {s}")
            total_failures += 1
            continue
        try:
            recomputed = compute_metrics_from_equity(str(equity), str(trades) if trades.exists() else None)
        except Exception as e:
            print(f"  Failed to recompute metrics: {e}")
            total_failures += 1
            continue

        mismatches = compare_dicts(data, recomputed)
        if mismatches:
            total_failures += 1
            print("  Mismatches detected:")
            for m in mismatches:
                print("   -", m)
            print("  Reference:", {k: data.get(k) for k in ('sharpe','sortino','max_drawdown','profit_factor','final_equity')})
            print("  Recalc   :", {k: recomputed.get(k) for k in ('sharpe','sortino','max_drawdown','profit_factor','final_equity')})
        else:
            print("  OK: metrics within tolerances.")

    if total_failures:
        print(f"Validation failures: {total_failures}")
        return 2
    print("All summary.json files validated successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
