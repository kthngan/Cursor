# Option 2 — start Lewis Schedule and smoke-test on PC
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
Set-Location $agent

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env — edit WORKSPACE_DIR and CURSOR_API_KEY before Option 3." -ForegroundColor Yellow
}

$envContent = Get-Content ".env" -Raw
if ($envContent -notmatch "WORKSPACE_DIR=.+" -or $envContent -match "WORKSPACE_DIR=\s*$") {
    $repoRoot = (Resolve-Path (Join-Path $root "..")).Path
    Add-Content ".env" "WORKSPACE_DIR=$repoRoot"
    Write-Host "Set WORKSPACE_DIR=$repoRoot"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\pip.exe" install -q -r requirements.txt

$token = "lewis-2026-test"
$match = Select-String -Path ".env" -Pattern '^\s*ACCESS_TOKEN=(.+)$' | Select-Object -First 1
if ($match -and $match.Matches.Groups[1].Value.Trim()) {
    $token = $match.Matches.Groups[1].Value.Trim()
}

$env:WORKSPACE_DIR = (Select-String -Path ".env" -Pattern '^\s*WORKSPACE_DIR=(.+)$' | ForEach-Object { $_.Matches.Groups[1].Value.Trim() })
if (-not $env:WORKSPACE_DIR) {
    $env:WORKSPACE_DIR = (Resolve-Path (Join-Path $root "..")).Path
}

Write-Host ""
Write-Host "=== Lewis Schedule — Option 2 test ===" -ForegroundColor Cyan
Write-Host "Starting server in this window..."
Write-Host "Browser:  http://127.0.0.1:8790"
Write-Host "Token:    $token"
Write-Host ""
Write-Host "After server starts, open another PowerShell and run:" -ForegroundColor Yellow
Write-Host "  cd $root"
Write-Host "  .\test-import.ps1"
Write-Host ""

Start-Process "http://127.0.0.1:8790"
& ".venv\Scripts\python.exe" server.py
