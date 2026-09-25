#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LAB_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_DIR/trainer/src:$REPO_DIR/worker/src:$LAB_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"
exec "${PYTHON:-python}" -m tool_policy_lab.cli "$@"
