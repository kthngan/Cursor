#!/usr/bin/env bash
# Keep GitHub Pages copy in sync with the standalone app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cp "$ROOT/lewis-schedule/standalone/index.html" "$ROOT/docs/index.html"
echo "Synced docs/index.html from lewis-schedule/standalone/index.html"
