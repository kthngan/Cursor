# Start Lewis Schedule web + agent locally (Windows PowerShell)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
Set-Location $agent

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created agent/.env — set CURSOR_API_KEY and ACCESS_TOKEN before importing screenshots."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".venv\Scripts\pip.exe" install -r requirements.txt
$env:WORKSPACE_DIR = (Resolve-Path (Join-Path $root "..")).Path
Write-Host "WORKSPACE_DIR=$env:WORKSPACE_DIR"
Write-Host "Open http://127.0.0.1:8790"
& ".venv\Scripts\python.exe" server.py
