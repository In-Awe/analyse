#!/usr/bin/env bash
# scripts/download_all_pairs.py
# Orchestrate downloading historic klines from Binance for many pairs.
# Usage examples:
#  python scripts/download_all_pairs.py --symbols BTCUSDT,ETHUSDT --months 84
#  python scripts/download_all_pairs.py --top 50 --months 84
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT"
export PYTHONPATH="${PY}:${PYTHONPATH:-}"

python - <<'PY'
import argparse, os, sys
from pathlib import Path
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--symbols", default="", help="Comma-separated symbols to download (e.g. BTCUSDT,ETHUSDT)")
parser.add_argument("--top", type=int, default=0, help="If >0, download top N symbols returned by Binance (quick filter on USDT pairs)")
parser.add_argument("--months", type=int, default=84, help="Months back to attempt (default=84 ~ 7 years)")
parser.add_argument("--api_key", default="", help="Optional API key for increased rate limits (not required).")
parser.add_argument("--out", default="artifacts/raw", help="Output directory")
args = parser.parse_args()

from tools.binance_public_fetcher import get_all_symbols, download_symbol_history

symbols = []
if args.symbols:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
elif args.top and args.top > 0:
    all_symbols = get_all_symbols()
    # Filter USDT pairs as likely relevant; pick top by alphabetical (no volume info without external calls)
    usdt = [s for s in all_symbols if s.endswith("USDT")]
    symbols = usdt[:args.top]
else:
    # default small set
    symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT"]

print("Will download symbols:", symbols)
for sym in symbols:
    print("Downloading:", sym)
    ok = download_symbol_history(sym, out_dir=args.out, months_back=args.months, api_key=args.api_key)
    print("Done", sym, "ok:", ok)
    # small sleep between symbols to be polite
    sleep(1.0)

print("All downloads attempted.")
PY
