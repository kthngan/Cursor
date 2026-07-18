#!/usr/bin/env bash
# Keep GitHub Pages copies in sync with the standalone app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cp "$ROOT/lewis-schedule/standalone/index.html" "$ROOT/docs/schedule.html"
echo "Synced docs/schedule.html from lewis-schedule/standalone/index.html"
