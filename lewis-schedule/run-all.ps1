# Lewis Schedule — one-click setup: Option 2 + Option 3
# Double-click run-all.bat or: powershell -ExecutionPolicy Bypass -File run-all.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
$repoRoot = (Resolve-Path (Join-Path $root "..")).Path

Write-Host ""
Write-Host "=== Lewis Schedule — full setup ===" -ForegroundColor Cyan
Write-Host ""

Set-Location $repoRoot
if (Test-Path ".git") {
    Write-Host "Pulling latest from main..."
    & git pull origin main 2>$null
}

Set-Location $agent
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

$envLines = @(Get-Content ".env")
$envMap = @{}
foreach ($line in $envLines) {
    if ($line -match '^\s*([^#=]+)=(.*)$') {
        $envMap[$Matches[1].Trim()] = $Matches[2].Trim()
    }
}

if (-not $envMap["WORKSPACE_DIR"]) {
    $envMap["WORKSPACE_DIR"] = $repoRoot
}
if (-not $envMap["ACCESS_TOKEN"]) {
    $envMap["ACCESS_TOKEN"] = "lulufeijai"
}
if (-not $envMap["HOST"]) {
    $envMap["HOST"] = "0.0.0.0"
}
if (-not $envMap["PORT"]) {
    $envMap["PORT"] = "8790"
}

$key = $envMap["CURSOR_API_KEY"]
if (-not $key -or $key -eq "cursor_your_key_here") {
    Write-Host ""
    Write-Host "Cursor API key required for screenshot import (Option 3)." -ForegroundColor Yellow
    Write-Host "Get one at: https://cursor.com/dashboard/integrations" -ForegroundColor Yellow
    Write-Host ""
    $key = Read-Host "Paste your CURSOR_API_KEY here (starts with cursor_)"
    if (-not $key) {
        Write-Host "No key entered — Option 2 only (grid + export)." -ForegroundColor Yellow
    } else {
        $envMap["CURSOR_API_KEY"] = $key.Trim()
    }
}

$newEnv = @()
foreach ($k in @("CURSOR_API_KEY", "ACCESS_TOKEN", "WORKSPACE_DIR", "HOST", "PORT")) {
    if ($envMap.ContainsKey($k) -and $envMap[$k]) {
        $newEnv += "$k=$($envMap[$k])"
    }
}
$newEnv | Set-Content ".env" -Encoding UTF8

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python environment (first run only)..."
    python -m venv .venv
}
Write-Host "Installing dependencies..."
& ".venv\Scripts\pip.exe" install -q -r requirements.txt
& ".venv\Scripts\pip.exe" install -q pillow 2>$null

$token = $envMap["ACCESS_TOKEN"]
$port = $envMap["PORT"]

# Stop any existing server on this port
Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

Write-Host "Starting server on port $port..."
$serverJob = Start-Job -ScriptBlock {
    param($agentPath, $workspaceDir)
    Set-Location $agentPath
    $env:WORKSPACE_DIR = $workspaceDir
    & ".venv\Scripts\python.exe" server.py
} -ArgumentList $agent, $repoRoot

Start-Sleep -Seconds 6

$health = $null
for ($i = 0; $i -lt 10; $i++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" `
            -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 5
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $health) {
    Write-Host "Server failed to start. Job output:" -ForegroundColor Red
    Receive-Job $serverJob
    exit 1
}

Write-Host ""
Write-Host "Option 2 — READY" -ForegroundColor Green
Write-Host "  Browser: http://127.0.0.1:$port"
Write-Host "  Token:   $token"
Write-Host "  Composer: $($health.composer_available)"
Write-Host ""

Start-Process "http://127.0.0.1:$port"

if ($health.composer_available) {
    Write-Host "Running Option 3 — screenshot import test..." -ForegroundColor Cyan
    Set-Location $root
    & "$root\test-import.ps1"
} else {
    Write-Host "Option 3 skipped — add CURSOR_API_KEY to agent\.env and run .\test-import.ps1" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Server is running in the background (Job $($serverJob.Id))." -ForegroundColor Cyan
Write-Host "To stop: Stop-Job $($serverJob.Id); Remove-Job $($serverJob.Id)" -ForegroundColor Gray
Write-Host "Or close this PowerShell window." -ForegroundColor Gray
