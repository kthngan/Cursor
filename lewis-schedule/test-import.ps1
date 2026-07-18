# Option 3 — test screenshot import via Composer (needs CURSOR_API_KEY in agent/.env)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
$fixture = Join-Path $root "fixtures\test-screenshot.png"

if (-not (Test-Path (Join-Path $agent ".env"))) {
    Write-Error "Missing agent\.env — run test-local.ps1 first or copy .env.example"
}

$keyLine = Select-String -Path (Join-Path $agent ".env") -Pattern '^\s*CURSOR_API_KEY=(.+)$' | Select-Object -First 1
$key = if ($keyLine) { $keyLine.Matches.Groups[1].Value.Trim() } else { "" }
if (-not $key -or $key -eq "cursor_your_key_here") {
    Write-Host "Add your Cursor API key to lewis-schedule\agent\.env:" -ForegroundColor Red
    Write-Host "  CURSOR_API_KEY=cursor_..."
    Write-Host "Get it from https://cursor.com/dashboard/integrations"
    exit 1
}

$token = "lewis-2026-test"
$tokLine = Select-String -Path (Join-Path $agent ".env") -Pattern '^\s*ACCESS_TOKEN=(.+)$' | Select-Object -First 1
if ($tokLine -and $tokLine.Matches.Groups[1].Value.Trim()) {
    $token = $tokLine.Matches.Groups[1].Value.Trim()
}

if (-not (Test-Path $fixture)) {
    Write-Host "Creating test screenshot (Swimming Thursday)..."
    python -c @"
from pathlib import Path
try:
    from PIL import Image, ImageDraw
    p = Path(r'$fixture')
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (400, 120), 'white')
    ImageDraw.Draw(img).text((20, 40), 'Swimming Thursday', fill='black')
    img.save(p)
    print('Created', p)
except ImportError:
    print('Install pillow: pip install pillow')
    raise
"@
}

Write-Host "Checking server..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8790/api/health" -Headers @{ Authorization = "Bearer $token" }
} catch {
    Write-Host "Server not running. Start it first:" -ForegroundColor Red
    Write-Host "  cd $root"
    Write-Host "  .\test-local.ps1"
    exit 1
}

if (-not $health.composer_available) {
    Write-Host "Server running but Composer not configured. Restart server after setting CURSOR_API_KEY." -ForegroundColor Red
    exit 1
}

Write-Host "Composer available. Loading template..." -ForegroundColor Green
$template = Invoke-RestMethod -Uri "http://127.0.0.1:8790/api/template" -Headers @{ Authorization = "Bearer $token" }

$schedule = @{
    week_start = $template.week_start
    caregivers = $template.caregivers
    activities = $template.activities
    slots = $template.slots
}

$bytes = [IO.File]::ReadAllBytes($fixture)
$b64 = [Convert]::ToBase64String($bytes)

Write-Host "Sending test screenshot to Composer..." -ForegroundColor Cyan
$body = @{
    week_start = $template.week_start
    schedule = $schedule
    image_base64 = $b64
    mime_type = "image/png"
} | ConvertTo-Json -Depth 10 -Compress

$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8790/schedule/import/start" `
    -Headers @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" } `
    -Body $body

Write-Host ""
Write-Host "=== Composer response ===" -ForegroundColor Cyan
Write-Host "Mode:    $($response.agent.mode)"
Write-Host "Message: $($response.agent.message)"
if ($response.agent.questions) {
    foreach ($q in $response.agent.questions) {
        Write-Host "Question: $($q.text)"
        Write-Host "  Choices: $($q.choices -join ', ')"
    }
}
if ($response.agent.patch) {
    Write-Host "Proposed patch:"
    $response.agent.patch | ConvertTo-Json
}
Write-Host ""
Write-Host "Thread ID: $($response.thread_id)"
Write-Host "Option 3 test complete. Use the web UI to answer questions and Apply." -ForegroundColor Green
