#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
AGENT="$ROOT/agent"
cd "$AGENT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created agent/.env — set CURSOR_API_KEY and ACCESS_TOKEN before importing screenshots."
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt
export WORKSPACE_DIR="$(cd "$ROOT/.." && pwd)"
echo "WORKSPACE_DIR=$WORKSPACE_DIR"
echo "Open http://127.0.0.1:8790"
python server.py
