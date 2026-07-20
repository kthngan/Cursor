#!/usr/bin/env bash
# Linux/macOS equivalent of run-all.ps1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
AGENT="$ROOT/agent"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
PORT="${PORT:-8790}"
TOKEN="${ACCESS_TOKEN:-lulufeijai}"

echo ""
echo "=== Lewis Schedule — full setup ==="
echo ""

cd "$REPO_ROOT"
git pull origin main 2>/dev/null || true

cd "$AGENT"
if [[ ! -f .env ]]; then
  cp .env.example .env
fi

# shellcheck disable=SC1091
source /dev/null
WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_ROOT}"
if grep -q '^WORKSPACE_DIR=' .env 2>/dev/null; then
  sed -i "s|^WORKSPACE_DIR=.*|WORKSPACE_DIR=$REPO_ROOT|" .env
else
  echo "WORKSPACE_DIR=$REPO_ROOT" >> .env
fi
grep -q '^ACCESS_TOKEN=' .env || echo "ACCESS_TOKEN=$TOKEN" >> .env
grep -q '^HOST=' .env || echo "HOST=127.0.0.1" >> .env
grep -q '^PORT=' .env || echo "PORT=$PORT" >> .env

KEY="$(grep '^CURSOR_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)"
if [[ -z "$KEY" || "$KEY" == "cursor_your_key_here" ]]; then
  if [[ -n "${CURSOR_API_KEY:-}" ]]; then
    if grep -q '^CURSOR_API_KEY=' .env; then
      sed -i "s|^CURSOR_API_KEY=.*|CURSOR_API_KEY=$CURSOR_API_KEY|" .env
    else
      echo "CURSOR_API_KEY=$CURSOR_API_KEY" >> .env
    fi
    KEY="$CURSOR_API_KEY"
  else
    echo "No CURSOR_API_KEY — Option 3 (import) will be skipped."
  fi
fi

if [[ ! -d .venv/bin ]]; then
  echo "Creating Python environment..."
  if python3 -m venv .venv 2>/dev/null; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -q -r requirements.txt
    pip install -q pillow 2>/dev/null || true
  else
    echo "Using system Python (venv unavailable)."
    pip3 install -q -r requirements.txt 2>/dev/null || true
    pip3 install -q pillow 2>/dev/null || true
    PYTHON=python3
  fi
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q -r requirements.txt
  pip install -q pillow 2>/dev/null || true
fi
PYTHON="${PYTHON:-python3}"

fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

export WORKSPACE_DIR="$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
echo "Starting server on port $PORT..."
python3 server.py &
SERVER_PID=$!
sleep 6

HEALTH=""
for _ in $(seq 1 10); do
  if HEALTH="$(curl -sf -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/api/health")"; then
    break
  fi
  sleep 2
done

if [[ -z "$HEALTH" ]]; then
  echo "Server failed to start."
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

echo ""
echo "Option 2 — READY"
echo "  URL:   http://127.0.0.1:$PORT"
echo "  Token: $TOKEN"
echo "  $HEALTH"
echo ""

if echo "$HEALTH" | grep -q '"composer_available":true'; then
  echo "Running Option 3 — screenshot import test..."
  cd "$ROOT"
  bash "$ROOT/test-import.sh"
else
  echo "Option 3 skipped — set CURSOR_API_KEY in agent/.env"
fi

echo ""
echo "Server PID: $SERVER_PID (kill $SERVER_PID to stop)"
