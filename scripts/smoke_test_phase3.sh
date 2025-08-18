#!/usr/bin/env bash
set -euo pipefail
# Quick smoke test for Phase II+III (small run)
CLEANED="data/cleaned/BTCUSD_1min.cleaned.csv"
FEATURES="data/features/technical.parquet"
FEATURES_WITH_TARGET="data/features/features_with_target.parquet"
MERGED="data/features/merged_features_with_target.parquet"

echo "[smoke] ensure cleaned CSV exists: $CLEANED"
if [ ! -f "$CLEANED" ]; then
  echo "ERROR: cleaned CSV missing ($CLEANED). Produce Phase I outputs first."
  exit 2
fi

echo "[smoke] build features (fast)"
python scripts/build_features.py --input "$CLEANED" --out "$FEATURES"

echo "[smoke] create target parquet"
python src/models/target.py --input "$CLEANED" --out "$FEATURES_WITH_TARGET"

echo "[smoke] merge features + target"
python - <<PY
import pandas as pd
fe = pd.read_parquet("$FEATURES")
tg = pd.read_parquet("$FEATURES_WITH_TARGET")
m = tg.merge(fe, on="timestamp", how="left")
m.to_parquet("$MERGED", index=False)
print("[smoke] merged rows:", len(m))
PY

echo "[smoke] training XGBoost small run (nrounds=10)"
python src/models/train_xgb.py --features "$MERGED" --out-dir "artifacts/models/xgb_smoke" --nrounds 10 --early-stopping-rounds 5

echo "[smoke] done. check artifacts/models/xgb_smoke for xgb_model.json and xgb_metrics.json"
ls -la artifacts/models/xgb_smoke || true
