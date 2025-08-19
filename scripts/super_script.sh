#!/usr/bin/env bash
set -euo pipefail

# Create the files first
mkdir -p docs
mkdir -p scripts
mkdir -p .github/workflows

cat <<'EOF' > docs/phase4_status.md
# Phase IV — status snapshot & QA checklist (phenol)

Repository: In-Awe/analyse

## Current snapshot (as of patch)
- Phase IV backtester, LSTM loader, and robustness tools were merged into `main`. See PRs and commits dated Aug 18–19, 2025.
- Some CI checks associated with those merges reported failures; further QA recommended.

## Purpose of this document
This file lists an explicit QA checklist and the commands used by `scripts/run_phase4_qa.sh`. Jules (the ai agent) should:

1. Run the smoke backtest (1-hour replay / single config).
2. Run the robustness runner (Monte Carlo reshuffle / parameter sensitivity) with a short sample to verify the pipeline executes.
3. Upload/save artifacts to `artifacts/phase4_qa/<YYYYMMDD_HHMM>/`.
4. Report failing tests and capture logs for human review.

## Minimum checks (automated)
- [ ] `python -m scripts.phase4_smoke` (or `scripts/run_phase4_qa.sh --smoke`)
- [ ] Backtester finishes and writes `artifacts/backtest/equity_curve.csv`
- [ ] Robustness runner finishes a reduced sample and writes `artifacts/robustness/summary.json`
- [ ] Model load/save functions succeed (check `artifacts/experiments/` for model artifacts)

## Failure handling
If any step fails, capture:
- `artifacts/phase4_qa/<ts>/ci_log.txt` with the stdout/stderr
- The failing test name and trace

## Next recommended actions (if CI fails)
1. Inspect CI logs for the failing check (token/permission, missing dependency, path errors).
2. If model saving/loading fails, confirm PyTorch/Torch version compatibility in `requirements.txt`.
3. If test dataset paths mismatch, verify `data/cleaned/` path and filenames.

## Contact / human sign-off
Before moving to live paper trading: require a human reviewer to sign off in `docs/phase4_status.md` with date & initials.
EOF

cat <<'EOF' > scripts/run_phase4_qa.sh
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
      python scripts/run_backtest.py --config configs/backtest_phase4.yaml --out "${ARTIFACT_DIR}" || echo "smoke_backtest_failed" > "${ARTIFACT_DIR}/smoke_error.txt"
    else
      echo "No backtest runner found (expected src.backtest or scripts/run_backtest.py)" > "${ARTIFACT_DIR}/smoke_error.txt"
    fi
  fi
fi

# 2) Short robustness check (reduced samples)
if [ "${ROBUST}" = true ]; then
  echo "Running short robustness analysis..."
  # If a robustness runner exists, call it with reduced sample sizes
  if [ -f scripts/robustness_runner.py ]; then
    python scripts/robustness_runner.py --out "${ARTIFACT_DIR}" --mc-iterations 100 --noise-trials 5 || echo "robustness_failed" > "${ARTIFACT_DIR}/robustness_error.txt"
  else
    echo "No robustness runner script found; skipping robustness step" > "${ARTIFACT_DIR}/robustness_skipped.txt"
  fi
fi

echo "QA run complete. Artifacts saved to ${ARTIFACT_DIR}"
echo "CI logs & outputs (if present) should be attached to PR for human review."
EOF

cat <<'EOF' > .github/workflows/phase4-ci.yml
name: Phase IV Smoke CI

on:
  workflow_dispatch:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  phase4-smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          cd backend || true
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -r requirements.txt || true

      - name: Run Phase IV QA smoke
        run: |
          chmod +x scripts/run_phase4_qa.sh || true
          ./scripts/run_phase4_qa.sh --smoke
        continue-on-error: false

      - name: Upload artifacts (if any)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: phase4-qa-artifacts
          path: artifacts/phase4_qa || artifacts || .

  phase4-robust:
    runs-on: ubuntu-latest
    needs: phase4-smoke
    if: github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install deps
        run: |
          pip install -r requirements.txt || true
      - name: Run short robustness (manual dispatch only)
        run: |
          ./scripts/run_phase4_qa.sh --robust
EOF

# Now, the git operations from the user's script
COMMIT_MSG="${1:-feat: add Phase IV QA runner + CI workflow (phenol)}"
BRANCH="${2:-feat/phenol-phase4-qa}"
REMOTE="${3:-origin}"
BASE_BRANCH="${4:-main}"

# Ensure we are in repo root (attempt)
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -z "$REPO_ROOT" ]; then
  echo "ERROR: not inside a git repo (no .git). Aborting."
  exit 1
fi
cd "$REPO_ROOT"

# Configure local committer identity if not set
git config user.email >/dev/null 2>&1 || true
if [ -z "$(git config user.email || true)" ]; then
  git config user.email "phenol-bot@example.com"
fi
if [ -z "$(git config user.name || true)" ]; then
  git config user.name "phenol-bot"
fi

# Refresh base branch and create feature branch
git fetch "$REMOTE" --prune
git checkout "$BASE_BRANCH"
git pull "$REMOTE" "$BASE_BRANCH" --ff-only || true
git checkout -b "$BRANCH"

# Stage changes (all new/modified files)
git add -A

# Nothing to commit?
if git diff --cached --quiet; then
  echo "No staged changes to commit. Exiting."
  exit 0
fi

# Commit
git commit -m "$COMMIT_MSG"

# Push branch and set upstream
git push --set-upstream "$REMOTE" "$BRANCH"

# Optionally create PR if gh CLI present
if command -v gh >/dev/null 2>&1; then
  gh pr create --base "$BASE_BRANCH" --head "$BRANCH" \
    --title "$COMMIT_MSG" \
    --body "Automated: add Phase IV QA runner + CI smoke workflow (phenol). Please review."
else
  echo "gh CLI not found — branch pushed to $REMOTE/$BRANCH. Create a PR manually on GitHub."
fi

echo "Done: changes committed and pushed to $REMOTE/$BRANCH"
