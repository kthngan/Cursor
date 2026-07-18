#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
AGENT="$ROOT/agent"
FIXTURE="$ROOT/fixtures/test-screenshot.png"
TOKEN="${ACCESS_TOKEN:-lewis-2026-test}"
PORT="${PORT:-8790}"

KEY="$(grep '^CURSOR_API_KEY=' "$AGENT/.env" 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)"
if [[ -z "$KEY" || "$KEY" == "cursor_your_key_here" ]]; then
  echo "Missing CURSOR_API_KEY in agent/.env"
  exit 1
fi

mkdir -p "$ROOT/fixtures"
if [[ ! -f "$FIXTURE" ]]; then
  python3 - <<'PY'
from PIL import Image, ImageDraw
from pathlib import Path
p = Path("/workspace/lewis-schedule/fixtures/test-screenshot.png")
img = Image.new("RGB", (400, 120), "white")
ImageDraw.Draw(img).text((20, 40), "Swimming Thursday", fill="black")
img.save(p)
print("Created", p)
PY
fi

TEMPLATE="$(curl -sf -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/api/template")"
B64="$(base64 -w0 "$FIXTURE")"

python3 - <<PY
import json, urllib.request
body = {
    "week_start": json.loads('''$TEMPLATE''')["week_start"],
    "schedule": json.loads('''$TEMPLATE'''),
    "image_base64": "$B64",
    "mime_type": "image/png",
}
req = urllib.request.Request(
    "http://127.0.0.1:$PORT/schedule/import/start",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": "Bearer $TOKEN",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.load(resp)
agent = data.get("agent", {})
print("Mode:", agent.get("mode"))
print("Message:", agent.get("message"))
for q in agent.get("questions") or []:
    print("Question:", q.get("text"))
    print("  Choices:", ", ".join(q.get("choices") or []))
if agent.get("patch"):
    print("Patch:", json.dumps(agent["patch"], indent=2))
print("Thread:", data.get("thread_id"))
PY
