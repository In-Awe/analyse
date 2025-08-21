#!/usr/bin/env python3
"""
scripts/replay_historical.py

Replay selected historical raw CSVs into a temporary raw dir and run training/backtests on them.

Use-case: after you've downloaded many months of data and want to re-run training/backtests
only on a subset of historical months for a specific pair to test model stability.

Example:
  python scripts/replay_historical.py --pair BTCUSDT --from 2020-01 --to 2021-12 --out-dir artifacts/replay --run-train

This will:
 - Find files in artifacts/raw matching pair and the months
 - Create a temp replay dir artifacts/replay/<runid>/raw and symlink or copy files there
 - Call scripts/train_on_downloads.py --raw <replay_raw> --limit 0
 - Save outputs to artifacts/replay/<runid>/
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import tempfile
import subprocess
import sys
import os
from datetime import datetime
import glob

RAW_DIR = Path("artifacts/raw")
REPLAY_ROOT = Path("artifacts/replay")

def parse_month(s: str):
    # accept YYYY-MM or YYYY-MM-DD
    parts = s.split("-")
    if len(parts) < 2:
        raise ValueError("Invalid month format, expected YYYY-MM")
    year = int(parts[0]); month = int(parts[1])
    return year, month

def month_key_from_filename(fname: str):
    # try to extract YYYY-MM from filename like BTCUSDT_1m_2024-06.csv
    import re
    m = re.search(r"(\d{4}-\d{2})", fname)
    return m.group(1) if m else None

def find_matching_files(pair: str, start_month: str, end_month: str):
    files = []
    for path in RAW_DIR.rglob(f"*{pair}*.csv"):
        mk = month_key_from_filename(path.name)
        if mk is None:
            continue
        if start_month <= mk <= end_month:
            files.append(path)
    files.sort()
    return files

def symlink_files_into(files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        target = dest_dir / p.name
        if not target.exists():
            try:
                os.symlink(p.resolve(), target)
            except Exception:
                # fallback to copy if symlink not allowed
                shutil.copy2(p, target)

def run_training_on_replay(replay_raw: Path, extra_args: list[str], out_dir: Path):
    cmd = [sys.executable, "scripts/train_on_downloads.py", "--raw", str(replay_raw)]
    if extra_args:
        cmd += extra_args
    env = os.environ.copy()
    env["PYTHONPATH"] = f'{os.getcwd()}:{env.get("PYTHONPATH","")}'
    print("Running training on replay with:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    return rc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", required=True, help="Symbol pair substring to match (e.g. BTCUSDT)")
    p.add_argument("--from", dest="from_month", required=True, help="Start month YYYY-MM")
    p.add_argument("--to", dest="to_month", required=True, help="End month YYYY-MM")
    p.add_argument("--out-dir", default=str(REPLAY_ROOT), help="Base output directory for replay results")
    p.add_argument("--run-train", action="store_true", help="After symlinking files, run training on replay set")
    p.add_argument("--train-args", default="", help="Extra args to pass to train_on_downloads.py")
    args = p.parse_args()

    start = args.from_month
    end = args.to_month
    pair = args.pair
    files = find_matching_files(pair, start, end)
    if not files:
        print("No files found for pair", pair, "in range", start, "to", end)
        return 2
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest_base = Path(args.out_dir) / run_id
    replay_raw = dest_base / "raw"
    symlink_files_into(files, replay_raw)
    print(f"Prepared replay raw dir with {len(list(replay_raw.glob('*.csv')))} files at {replay_raw}")
    if args.run_train:
        extra = []
        if args.train_args:
            extra = args.train_args.split()
        rc = run_training_on_replay(replay_raw, extra, dest_base)
        print("Training return code:", rc)
    else:
        print("Replay prepared; run training manually with --run-train or call scripts/train_on_downloads.py with --raw", replay_raw)

if __name__ == "__main__":
    main()
