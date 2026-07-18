#!/usr/bin/env bash
# Create a minimal test screenshot for import testing
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURE="$DIR/fixtures/test-screenshot.png"
mkdir -p "$DIR/fixtures"
python3 - <<'PY'
from pathlib import Path
try:
    from PIL import Image, ImageDraw
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow"])
    from PIL import Image, ImageDraw
p = Path("/workspace/lewis-schedule/fixtures/test-screenshot.png")
p.parent.mkdir(parents=True, exist_ok=True)
img = Image.new("RGB", (400, 120), "white")
ImageDraw.Draw(img).text((20, 40), "Swimming Thursday", fill="black")
img.save(p)
print(f"Created {p}")
PY
