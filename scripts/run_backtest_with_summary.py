#!/usr/bin/env python3
"""
scripts/run_backtest_with_summary.py

Wrapper that runs the existing backtest runner (scripts/run_backtest.py), waits for it
to complete, then invokes tools.write_summary_atomic.write_summary_for_equity (via main)
to ensure summary.json files are present and written atomically.

Usage:
    python scripts/run_backtest_with_summary.py [<args forwarded to scripts/run_backtest.py>]

This wrapper returns the backtest return code if non-zero. If the backtest completes
successfully, the wrapper will attempt to create/update summary.json files; failures to
write summaries will not cause a non-zero exit (but are printed to stdout/stderr).
"""
from __future__ import annotations
import subprocess
import sys
import os

# Ensure the repository root is on PYTHONPATH so we can import tools.* modules.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def run_backtest_forward_args() -> int:
    # Prefer existing script location: scripts/run_backtest.py
    backtest_script = os.path.join(ROOT, "scripts", "run_backtest.py")
    if not os.path.exists(backtest_script):
        # fallback: try src/backtest/backtester.py
        cand = os.path.join(ROOT, "src", "backtest", "backtester.py")
        if os.path.exists(cand):
            backtest_script = cand
        else:
            print("ERROR: could not find scripts/run_backtest.py or src/backtest/backtester.py to execute.", file=sys.stderr)
            return 2
    cmd = [sys.executable, backtest_script] + sys.argv[1:]
    print("Running backtest command:", " ".join(cmd))
    rc = subprocess.call(cmd)
    return rc

def run_atomic_writer():
    try:
        # Prefer programmatic call
        from tools.write_summary_atomic import main as writer_main
        print("Running atomic summary writer via tools.write_summary_atomic.main()")
        writer_rc = writer_main()
        print("Atomic writer exit code:", writer_rc)
    except Exception as e:
        print("Falling back to subprocess invocation of tools/write_summary_atomic.py due to:", e)
        try:
            rc = subprocess.call([sys.executable, os.path.join(ROOT, "tools", "write_summary_atomic.py")])
            print("Subprocess writer rc:", rc)
        except Exception as e2:
            print("Failed to run atomic writer:", e2)

def main():
    rc = run_backtest_forward_args()
    if rc != 0:
        print("Backtest failed with rc", rc, "; skipping atomic summary writer.")
        sys.exit(rc)
    # run the atomic writer to ensure summary.json files exist
    run_atomic_writer()
    # success
    sys.exit(0)

if __name__ == "__main__":
    main()
