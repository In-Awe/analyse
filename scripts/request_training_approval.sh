#!/usr/bin/env bash
set -euo pipefail

# Simple helper to create a PR/checklist file for human training approval.
# This command does not modify configs/global.yaml automatically.
OUT=doc/training_approval_request.md
mkdir -p doc
cat > "${OUT}" <<'EOF'
# Training Start Request

This file records a request to enable model training for the repository.

Checklist (all must be verified and performed by a human reviewer):

- [ ] Code merged to `main` for system, features, backtest modules.
- [ ] CI green (unit-tests + replay-smoke).
- [ ] Artifacts present: cleaned data, feature parquet, backtest summary, monitoring configured.
- [ ] Risk limits configured in `configs/risk_limits.yaml`.
- [ ] Monitoring & alerting smoke-tested.
- [ ] `HUMAN_APPROVAL_TOKEN` entry created in the secure vault.
- [ ] A human has reviewed final_report.pdf and given explicit approval.

Once these are complete, set `TRAINING_ENABLED: true` in `configs/global.yaml` via a secure PR or vault action.

Approver notes:

_Add sign-off comment here with date and reviewer identity._
EOF

echo "Created ${OUT}. Create a PR for review or attach it to an approval issue."
exit 0
