#!/usr/bin/env bash
# Set up the environment for train_tool_server on EACH node.
#
# Usage (run from train_tool_server repo root):
#   bash scripts/setup_env.sh
#
# Assumes geo_edit sibling repo lives at ../geo_edit (relative to repo root).
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "[setup_env] repo root = $REPO_ROOT"

# === Step 1: pinned third-party deps from requirements.txt ===
pip install -r requirements.txt

# === Step 2: editable installs (no deps - already resolved above) ===
pip install -e . --no-deps

GEO_EDIT_DIR="$(cd "$REPO_ROOT/../geo_edit" 2>/dev/null && pwd || echo "")"
if [ -z "$GEO_EDIT_DIR" ] || [ ! -d "$GEO_EDIT_DIR" ]; then
    echo "[setup_env] ERROR: sibling repo ../geo_edit not found at $REPO_ROOT/../geo_edit" >&2
    exit 1
fi
pip install -e "$GEO_EDIT_DIR" --no-deps

echo "[setup_env] done. Verify with: python -m train_tool_server.server --help"
