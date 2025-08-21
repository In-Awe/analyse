import os
import csv
import argparse
import subprocess
import yaml

RAW_DIR = "data/raw"
MANIFEST = os.path.join(RAW_DIR, "manifest.csv")


def run_pipeline(entry, outdir):
    """
    Runs clean -> feature -> backtest for a single CSV.
    Assumes existing CLI scripts handle these steps.
    """
    fname = entry["filename"]
    symbol = entry["symbol"]
    ym = f"{entry['year']}-{entry['month']}"

    print(f"\n[RUN] {symbol} {ym} ({fname})")
    os.makedirs(outdir, exist_ok=True)

    cleaned_csv_path = os.path.join(outdir, "cleaned.csv")
    features_path = os.path.join(outdir, "features.parquet")
    imputation_log_path = os.path.join(outdir, "imputation_log.csv")

    cmds = [
        ["python", "scripts/clean_impute.py", "--input", os.path.join(RAW_DIR, fname), "--output-csv", cleaned_csv_path, "--output-log", imputation_log_path],
        ["python", "scripts/build_features.py", "--input", cleaned_csv_path, "--out", features_path],
    ]

    for cmd in cmds:
        print("  ->", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print("  !! Pipeline step failed:", e)
            return False

    # Create custom config for backtest
    try:
        with open("configs/trade_logic.yaml") as f:
            trade_logic_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(" !! trade_logic.yaml not found, cannot run backtest")
        return False

    if 'paths' not in trade_logic_cfg or trade_logic_cfg['paths'] is None:
        trade_logic_cfg['paths'] = {}

    trade_logic_cfg['paths']['cleaned_csv'] = cleaned_csv_path
    trade_logic_cfg['paths']['features_parquet'] = features_path
    trade_logic_cfg['paths']['outputs_dir'] = outdir

    custom_config_path = os.path.join(outdir, "trade_logic.yaml")
    with open(custom_config_path, 'w') as f:
        yaml.dump(trade_logic_cfg, f)

    cmd_backtest = ["python", "scripts/run_backtest.py", "--config", custom_config_path]
    print("  ->", " ".join(cmd_backtest))
    try:
        subprocess.check_call(cmd_backtest)
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
