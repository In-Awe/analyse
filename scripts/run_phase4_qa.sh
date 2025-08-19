#!/usr/bin/env bash
set -euo pipefail

# Small QA runner for Phase IV smoke and robustness checks.
# Usage:
#   ./scripts/run_phase4_qa.sh            # runs smoke + short robustness
#   ./scripts/run_phase4_qa.sh --smoke    # only smoke
#   ./scripts/run_phase4_qa.sh --robust   # only robustness (short)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARTIFACT_DIR="${ROOT}/artifacts/phase4_qa/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ARTIFACT_DIR}"

echo "Phase IV QA run started: artifacts -> ${ARTIFACT_DIR}"

SMOKE=true
ROBUST=true
if [ "${1:-}" = "--smoke" ]; then
  ROBUST=false
fi
if [ "${1:-}" = "--robust" ]; then
  SMOKE=false
fi

cd "${ROOT}"

# 1) Smoke backtest (short)
if [ "${SMOKE}" = true ]; then
  echo "Running smoke backtest..."
  # attempt to use existing backtest runner if present
  if python -c "import importlib, pkgutil; assert pkgutil.find_loader('src.backtest') is not None" >/dev/null 2>&1; then
    python -c "from src.backtest import backtest_cli; backtest_cli.run_smoke('${ARTIFACT_DIR}')" || { echo "Smoke backtest (module) failed" > "${ARTIFACT_DIR}/smoke_error.txt"; }
  else
    # fallback: try existing top-level script if present
    if [ -f scripts/run_backtest.py ]; then
      python scripts/run_backtest.py --data data/cleaned/BTCUSD_1min.cleaned.csv --config configs/backtest.yaml --out "${ARTIFACT_DIR}" || echo "smoke_backtest_failed" > "${ARTIFACT_DIR}/smoke_error.txt"
    else
      echo "No backtest runner found (expected src.backtest or scripts/run_backtest.py)" > "${ARTIFACT_DIR}/smoke_error.txt"
    fi
  fi
fi

# 2) Short robustness check (reduced samples)
if [ "${ROBUST}" = true ]; then
  echo "Running short robustness analysis..."
  # If a robustness runner exists, call it with reduced sample sizes
  if [ -f scripts/run_robustness.py ]; then
    python scripts/run_robustness.py --out "${ARTIFACT_DIR}" --mc-iterations 100 --noise-trials 5 || echo "robustness_failed" > "${ARTIFACT_DIR}/robustness_error.txt"
  else
    echo "No robustness runner script found; skipping robustness step" > "${ARTIFACT_DIR}/robustness_skipped.txt"
  fi
fi

echo "QA run complete. Artifacts saved to ${ARTIFACT_DIR}"
echo "CI logs & outputs (if present) should be attached to PR for human review."
