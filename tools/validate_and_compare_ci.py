#!/usr/bin/env python3
"""
tools/validate_and_compare_ci.py

CI helper to find equity/trades CSVs, run validate_metrics.py, and compare to a summary.json if present.
Exits with code 1 on validation errors or metric mismatches.

- Finds the most recent `equity_curve.csv` and `trades.csv` in `artifacts/`.
- Runs `validate_metrics.py` on them.
- If `summary.json` is present, compares generated metrics to its values.
- If not, prints metrics and exits cleanly.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

def find_latest_csv(directory, pattern="equity_curve.csv"):
    base = Path(directory)
    files = list(base.rglob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def run_validation(equity_csv, trades_csv):
    cmd = [sys.executable, "tools/validate_metrics.py", str(equity_csv)]
    if trades_csv:
        cmd.append(str(trades_csv))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)

def compare_metrics(gen, summ):
    errors = 0
    for k, v_gen in gen.items():
        if k in summ:
            v_summ = summ[k]
            # simple float comparison with tolerance
            if isinstance(v_gen, float) and abs(v_gen - v_summ) > 1e-6:
                print(f"Mismatch: {k} | generated={v_gen:.6f} | summary={v_summ:.6f}", file=sys.stderr)
                errors += 1
    return errors

def main():
    artifacts_dir = "artifacts"
    equity_csv = find_latest_csv(artifacts_dir, "equity_curve.csv")
    if not equity_csv:
        print("No equity_curve.csv found in artifacts/. Exiting.", file=sys.stderr)
        sys.exit(1)
    trades_csv = find_latest_csv(artifacts_dir, "trades.csv")
    summary_json = equity_csv.parent / "summary.json"

    print("Validating:", equity_csv)
    if trades_csv:
        print("Using trades:", trades_csv)

    try:
        metrics = run_validation(equity_csv, trades_csv)
        print("Generated metrics:\n", json.dumps(metrics, indent=2))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print("Failed to run validate_metrics.py:", e, file=sys.stderr)
        if hasattr(e, 'stderr'):
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

    if summary_json.exists():
        print("Comparing to:", summary_json)
        summary = json.loads(summary_json.read_text())
        errors = compare_metrics(metrics, summary)
        if errors > 0:
            print(f"Found {errors} metric mismatches.", file=sys.stderr)
            sys.exit(1)
        else:
            print("Metrics match summary.json. OK.")
    else:
        print("No summary.json found to compare against. OK.")

if __name__ == "__main__":
    main()
