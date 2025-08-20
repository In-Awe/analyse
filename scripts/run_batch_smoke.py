import os
import csv
import argparse
import subprocess

RAW_DIR = "data/raw"
MANIFEST = os.path.join(RAW_DIR, "manifest.csv")


def run_pipeline(entry, outdir):
    """
    Runs clean -> feature -> target -> backtest for a single CSV.
    Assumes existing CLI scripts handle these steps.
    """
    fname = entry["filename"]
    symbol = entry["symbol"]
    ym = f"{entry['year']}-{entry['month']}"

    print(f"\n[RUN] {symbol} {ym} ({fname})")
    os.makedirs(outdir, exist_ok=True)

    # Adjust to actual pipeline entry points as needed
    cmds = [
        ["python", "scripts/run_clean.py", "--input", os.path.join(RAW_DIR, fname), "--outdir", outdir],
        ["python", "scripts/run_features.py", "--indir", outdir, "--outdir", outdir],
        ["python", "scripts/run_target.py", "--indir", outdir, "--outdir", outdir],
        ["python", "scripts/run_backtest.py", "--indir", outdir, "--outdir", outdir],
    ]

    for cmd in cmds:
        print("  ->", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print("  !! Pipeline step failed:", e)
            return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1, help="Number of manifest rows to run")
    parser.add_argument("--all", action="store_true", help="Run all entries")
    args = parser.parse_args()

    if not os.path.exists(MANIFEST):
        raise FileNotFoundError("Manifest not found. Run catalog_raw_files.py first.")

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    if not args.all:
        rows = rows[: args.limit]

    for row in rows:
        outdir = os.path.join("artifacts", "batch", f"{row['symbol']}_{row['year']}-{row['month']}")
        ok = run_pipeline(row, outdir)
        if not ok:
            print(f"[FAIL] {row}")
        else:
            print(f"[OK] {row}")


if __name__ == "__main__":
    main()
