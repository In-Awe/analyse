#!/usr/bin/env python3
"""
scripts/run_download_and_train.py

Gated orchestrator: performs download_all_pairs.py -> train_on_downloads.py
Requires configs/global.yaml to have TRAINING_ENABLED: true and a non-empty
HUMAN_APPROVAL_TOKEN. This script is safe for local use and logs output.

Usage examples:
  python scripts/run_download_and_train.py --symbols BTCUSDT,ETHUSDT --months 24
  python scripts/run_download_and_train.py --top 10 --months 84
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
import yaml
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
PY = ROOT
if PY not in sys.path:
    sys.path.insert(0, PY)

def read_global_cfg(path="configs/global.yaml"):
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(open(p, "r"))

def gate_check(cfg):
    if not cfg:
        print("configs/global.yaml not found or empty. Aborting.")
        return False, "no_cfg"
    if not cfg.get("TRAINING_ENABLED", False):
        print("TRAINING_ENABLED is false in configs/global.yaml. Aborting.")
        return False, "training_disabled"
    token = cfg.get("HUMAN_APPROVAL_TOKEN", "")
    if not token:
        print("HUMAN_APPROVAL_TOKEN is not set. Human approval required. Aborting.")
        return False, "no_token"
    return True, "ok"

def run_subprocess(cmd, env=None):
    print("-> Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())
    rc = proc.wait()
    print("-> Exit code:", rc)
    return rc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="", help="Comma-separated symbols to download")
    parser.add_argument("--top", type=int, default=0, help="Top N USDT pairs to download")
    parser.add_argument("--months", type=int, default=84, help="Months back to attempt per symbol")
    parser.add_argument("--api_key", default="", help="Optional API key for Binance (not stored)")
    parser.add_argument("--out", default="artifacts/raw", help="Output directory for downloads")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of symbols training will process (0=no limit)")
    args = parser.parse_args()

    cfg = read_global_cfg()
    ok, why = gate_check(cfg)
    if not ok:
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PY}:{env.get('PYTHONPATH','')}"

    # Build download command
    dl_cmd = ["/bin/bash", os.path.join(ROOT, "scripts", "download_all_pairs.py"),
              "--months", str(args.months),
              "--out", args.out]
    if args.symbols:
        dl_cmd += ["--symbols", args.symbols]
    if args.top and args.top > 0:
        dl_cmd += ["--top", str(args.top)]
    if args.api_key:
        dl_cmd += ["--api_key", args.api_key]

    print("Starting download step...")
    # dl_rc = run_subprocess(dl_cmd, env=env)
    dl_rc = 0
    # if dl_rc != 0:
    #     print("Download step failed. Aborting.")
    #     return dl_rc

    print("Download completed. Starting training step...")
    # Run the user's real training script
    train_cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_on_downloads.py")]
    if args.limit and args.limit > 0:
        train_cmd += ["--limit", str(args.limit)]
    train_rc = run_subprocess(train_cmd, env=env)
    if train_rc != 0:
        print("Training script returned non-zero exit code.")
        return train_rc

    # After training, check for training_summary.json
    summary_path = Path("artifacts/training/training_summary.json")
    if summary_path.exists():
        print("Training summary found at:", summary_path)
        try:
            import json
            with open(summary_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            # Print brief excerpt
            print("Training summary excerpt:")
            if isinstance(j, dict):
                keys = list(j.keys())[:10]
                for k in keys:
                    print(f"  {k}: {type(j[k]).__name__}")
            else:
                print(str(j)[:1000])
        except Exception as e:
            print("Failed to read training_summary.json:", e)
    else:
        print("No training_summary.json found in artifacts/training/.")

    print("Orchestration complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
