#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cp "$ROOT/lewis-schedule/standalone/index.html" "$ROOT/docs/schedule.html"
cp "$ROOT/lewis-schedule/standalone/index.html" "$ROOT/docs/v6.html"
echo "Synced docs/schedule.html and docs/v6.html"
