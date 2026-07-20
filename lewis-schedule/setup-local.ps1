# One-time local server setup (Windows)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
$tools = Join-Path $root "tools"
$repoRoot = (Resolve-Path (Join-Path $root "..")).Path

Write-Host ""
Write-Host "=== Lewis Schedule - local setup ===" -ForegroundColor Cyan
Write-Host ""

Set-Location $agent

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

$envPath = Join-Path $agent ".env"
$lines = @(Get-Content $envPath)
$updated = $false
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s*WORKSPACE_DIR=\s*$') {
        $lines[$i] = "WORKSPACE_DIR=$repoRoot"
        $updated = $true
    }
    if ($lines[$i] -match '^\s*HOST=\s*$') {
        $lines[$i] = "HOST=0.0.0.0"
        $updated = $true
    }
    if ($lines[$i] -match '^\s*PORT=\s*$') {
        $lines[$i] = "PORT=8790"
        $updated = $true
    }
}
if ($updated) {
    $lines | Set-Content $envPath -Encoding UTF8
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

Write-Host "Installing Python dependencies..."
& ".venv\Scripts\pip.exe" install -q -r requirements.txt
try { & ".venv\Scripts\pip.exe" install -q pillow } catch { }

New-Item -ItemType Directory -Force -Path $tools | Out-Null
$cloudflared = Join-Path $tools "cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    Write-Host "Downloading cloudflared (internet tunnel)..."
    Invoke-WebRequest `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $cloudflared -UseBasicParsing
}

Write-Host "Configuring Windows Firewall rule for port 8790..."
$ruleName = "Lewis Schedule 8790"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if (-not $existing) {
    try {
        New-NetFirewallRule -DisplayName $ruleName `
            -Direction Inbound -Protocol TCP -LocalPort 8790 -Action Allow `
            -Profile Private, Domain | Out-Null
        Write-Host "Firewall rule added (Private + Domain networks)." -ForegroundColor Green
    } catch {
        Write-Host "Could not add firewall rule (run PowerShell as Admin):" -ForegroundColor Yellow
        Write-Host $_.Exception.Message
    }
} else {
    Write-Host "Firewall rule already exists." -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host '  1. Add CURSOR_API_KEY to agent\.env (for screenshot import)'
Write-Host '  2. Same Wi-Fi:     .\start.ps1'
Write-Host '  3. Over internet:  .\start-remote.ps1'
Write-Host ""
