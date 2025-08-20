#!/usr/bin/env python3
"""
tools/write_summary_atomic.py

Write (or rewrite) summary.json atomically for every discovered equity_curve.csv
under artifacts/**.

This file exposes a small programmatic API (write_summary_for_equity) so other
scripts can call it directly after running a backtest.

Why this exists:
  - Some sandboxes/overlays buffer or lose small files written late in a run.
  - We avoid partial/truncated writes with atomic replace + fsync.
  - CI then uses validate_and_compare_ci.py to compare "summary.json" against
    recomputed values, so we want these files to be reliably present.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Dict
import sys

# ensure repo root is on path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try to import compute helper (existing in repo)
try:
    # compute_metrics_from_equity(equity_csv: str, trades_csv: Optional[str]) -> Dict[str, float]
    from tools.validate_metrics import compute_metrics_from_equity
except Exception as e:
    # If module import fails, provide a clear message when the script is executed.
    compute_metrics_from_equity = None  # type: ignore

def _atomic_write_json(path: Path, data: Dict) -> None:
    """
    Write JSON to `path` atomically:
      - write to path.with_suffix('.tmp')
      - flush + fsync
      - os.replace(tmp, path)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Ensure compact+stable floats; indent aids debugging
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _compute_summary(equity_csv: Path, trades_csv: Optional[Path]) -> Dict:
    if compute_metrics_from_equity is None:
        raise RuntimeError("compute_metrics_from_equity not available; ensure tools/validate_metrics.py is on PYTHONPATH")
    return compute_metrics_from_equity(str(equity_csv), str(trades_csv) if trades_csv and trades_csv.exists() else None)

def write_summary_for_equity(equity_csv_path: str | Path, trades_csv_path: Optional[str | Path] = None) -> Path:
    """
    Compute metrics for a given equity_curve.csv and write summary.json next to it atomically.
    Returns the Path of the written summary.json.

    equity_csv_path: path to equity_curve.csv
    trades_csv_path: optional path to trades.csv (if not provided, sibling trades.csv will be used if present)
    """
    equity = Path(equity_csv_path)
    if not equity.exists():
        raise FileNotFoundError(f"Equity file not found: {equity}")
    trades = Path(trades_csv_path) if trades_csv_path else equity.parent / "trades.csv"
    metrics = _compute_summary(equity, trades if trades.exists() else None)
    summary = equity.parent / "summary.json"
    _atomic_write_json(summary, metrics)
    return summary

def main() -> int:
    """
    CLI entrypoint. Scans artifacts/ for equity_curve.csv files and writes summary.json
    next to each one. Returns exit code 0 on success.
    """
    if compute_metrics_from_equity is None:
        print("ERROR: tools.validate_metrics.compute_metrics_from_equity not importable. "
              "Ensure tools/validate_metrics.py is available and PYTHONPATH includes the repo root.", file=sys.stderr)
        return 2

    artifacts = Path("artifacts")
    if not artifacts.exists():
        print("No artifacts/ directory found; nothing to do.")
        return 0
    found = list(artifacts.rglob("equity_curve.csv"))
    if not found:
        print("No equity_curve.csv found under artifacts/; nothing to do.")
        return 0
    wrote = 0
    for equity in sorted(found):
        out_dir = equity.parent
        summary = out_dir / "summary.json"
        trades = out_dir / "trades.csv"
        try:
            metrics = _compute_summary(equity, trades if trades.exists() else None)
        except Exception as e:
            print(f"[write_summary_atomic] Failed to compute metrics for {equity}: {e}")
            continue
        try:
            _atomic_write_json(summary, metrics)
            wrote += 1
            print(f"[write_summary_atomic] Wrote {summary}")
        except Exception as e:
            print(f"[write_summary_atomic] Failed to write {summary}: {e}")
            continue
    if wrote == 0:
        print("No summaries written.")
    else:
        print(f"Wrote/updated {wrote} summary.json file(s).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
