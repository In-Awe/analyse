#!/usr/bin/env python3
"""
Write (or rewrite) summary.json atomically for every discovered equity_curve.csv
under artifacts/**.

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

try:
    from tools.validate_metrics import compute_metrics_from_equity
except Exception as e:
    print("ERROR: tools.validate_metrics not importable. Ensure it exists and is installable.", file=sys.stderr)
    raise

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
    return compute_metrics_from_equity(str(equity_csv), str(trades_csv) if trades_csv and trades_csv.exists() else None)

def main() -> int:
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
