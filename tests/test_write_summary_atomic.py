#!/usr/bin/env python3
"""
tests/test_write_summary_atomic.py

Simple unit test for tools.write_summary_atomic. It creates a temporary artifacts
tree with a mock equity_curve.csv and trades.csv, runs the writer, and asserts
summary.json exists and contains expected keys.
"""
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import shutil
import sys

# ensure repo root is on path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_summary_atomic import write_summary_for_equity

def make_mock_equity_and_trades(tmpdir: Path):
    run_dir = tmpdir / "artifacts" / "fake_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    equity = run_dir / "equity_curve.csv"
    trades = run_dir / "trades.csv"
    # create a tiny equity curve
    df_eq = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="T"),
        "equity": [1000.0, 1010.0, 1020.0]
    })
    df_eq.to_csv(equity, index=False)
    # create simple trades.csv with pnl column
    df_tr = pd.DataFrame({"pnl": [10.0, 10.0]})
    df_tr.to_csv(trades, index=False)
    return equity, trades

def test_write_summary_atomic(tmp_path):
    equity, trades = make_mock_equity_and_trades(tmp_path)
    # call the function under test
    summary_path = write_summary_for_equity(equity, trades)
    assert summary_path.exists(), "summary.json was not written"
    content = json.loads(summary_path.read_text(encoding="utf-8"))
    # check presence of common keys
    for k in ("final_equity", "sharpe", "sortino", "max_drawdown"):
        assert k in content, f"{k} missing from summary.json"
    # cleanup
    shutil.rmtree(tmp_path / "artifacts", ignore_errors=True)
