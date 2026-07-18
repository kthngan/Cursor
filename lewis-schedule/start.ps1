# Start Lewis Schedule web + agent locally (Windows PowerShell)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
Set-Location $agent

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Created agent\.env — review ACCESS_TOKEN and WORKSPACE_DIR before continuing."
    Write-Host ""
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".venv\Scripts\pip.exe" install -q -r requirements.txt
$repoRoot = (Resolve-Path (Join-Path $root "..")).Path
if (-not $env:WORKSPACE_DIR) {
    $env:WORKSPACE_DIR = $repoRoot
}

$token = "lewis-2026-test"
if (Test-Path ".env") {
    $match = Select-String -Path ".env" -Pattern '^\s*ACCESS_TOKEN=(.+)$' | Select-Object -First 1
    if ($match -and $match.Matches.Groups[1].Value.Trim()) {
        $token = $match.Matches.Groups[1].Value.Trim()
    }
}

Write-Host ""
Write-Host "=== Lewis Schedule ===" -ForegroundColor Cyan
Write-Host "WORKSPACE_DIR=$env:WORKSPACE_DIR"
Write-Host "PC browser:     http://127.0.0.1:8790"
Write-Host "Access token:   $token"
Write-Host ""

$addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -ExpandProperty IPAddress -Unique

if ($addresses) {
    Write-Host "iPhone (same Wi-Fi):" -ForegroundColor Green
    foreach ($ip in $addresses) {
        Write-Host "  http://${ip}:8790"
    }
} else {
    Write-Host "Could not detect LAN IP — run 'ipconfig' and use your Wi-Fi IPv4 address."
}

Write-Host ""
Write-Host "See SETUP-IPHONE.md for full iPhone steps."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

& ".venv\Scripts\python.exe" server.py
