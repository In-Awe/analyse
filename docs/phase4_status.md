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
