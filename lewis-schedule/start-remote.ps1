# Start Lewis Schedule + Cloudflare quick tunnel (works over the internet)
# Usage: powershell -ExecutionPolicy Bypass -File start-remote.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"
$tools = Join-Path $root "tools"
$cloudflared = Join-Path $tools "cloudflared.exe"
$repoRoot = (Resolve-Path (Join-Path $root "..")).Path

function Read-DotEnvValue {
    param([string]$Path, [string]$Key, [string]$Default = "")
    if (-not (Test-Path $Path)) { return $Default }
    $match = Select-String -Path $Path -Pattern "^\s*$([regex]::Escape($Key))=(.*)$" | Select-Object -First 1
    if ($match -and $match.Matches.Groups[1].Value.Trim()) {
        return $match.Matches.Groups[1].Value.Trim()
    }
    return $Default
}

function Stop-PortListener {
    param([int]$Port)
    Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

if (-not (Test-Path (Join-Path $agent ".venv"))) {
    & (Join-Path $root "setup-local.ps1")
}

if (-not (Test-Path $cloudflared)) {
    & (Join-Path $root "setup-local.ps1")
}

Set-Location $agent
$envPath = Join-Path $agent ".env"
$token = Read-DotEnvValue -Path $envPath -Key "ACCESS_TOKEN" -Default "lulufeijai"
$port = [int](Read-DotEnvValue -Path $envPath -Key "PORT" -Default "8790")
$env:WORKSPACE_DIR = Read-DotEnvValue -Path $envPath -Key "WORKSPACE_DIR" -Default $repoRoot

Stop-PortListener -Port $port

Write-Host ""
Write-Host "=== Lewis Schedule (local + internet) ===" -ForegroundColor Cyan
Write-Host "Access token: $token"
Write-Host ""

$serverJob = Start-Job -ScriptBlock {
    param($AgentPath, $WorkspaceDir)
    Set-Location $AgentPath
    $env:WORKSPACE_DIR = $WorkspaceDir
    & ".venv\Scripts\python.exe" server.py 2>&1
} -ArgumentList $agent, $env:WORKSPACE_DIR

$healthy = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" `
            -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 3
        $healthy = $true
        break
    } catch {
        # wait for server
    }
}

if (-not $healthy) {
    Write-Host "Server failed to start:" -ForegroundColor Red
    Receive-Job $serverJob
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Local PC:  http://127.0.0.1:$port" -ForegroundColor Green

$addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -ExpandProperty IPAddress -Unique
if ($addresses) {
    Write-Host "Same Wi-Fi:" -ForegroundColor Green
    foreach ($ip in $addresses) {
        Write-Host "  http://${ip}:$port"
    }
}

Write-Host ""
Write-Host "Starting Cloudflare tunnel (public HTTPS URL)..." -ForegroundColor Cyan
Write-Host "Leave this window open. Press Ctrl+C to stop server and tunnel."
Write-Host ""

$tunnelLog = Join-Path $env:TEMP "lewis-schedule-tunnel.log"
if (Test-Path $tunnelLog) { Remove-Item $tunnelLog -Force }

$tunnelJob = Start-Job -ScriptBlock {
    param($Cloudflared, $Port, $LogPath)
    & $Cloudflared tunnel --url "http://127.0.0.1:$Port" --no-autoupdate 2>&1 |
        Tee-Object -FilePath $LogPath
} -ArgumentList $cloudflared, $port, $tunnelLog

$publicUrl = $null
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
        $log = Get-Content $tunnelLog -Raw -ErrorAction SilentlyContinue
        if ($log -match '(https://[a-z0-9-]+\.trycloudflare\.com)') {
            $publicUrl = $Matches[1]
            break
        }
    }
}

if ($publicUrl) {
    Write-Host "Internet (iPhone anywhere):" -ForegroundColor Green
    Write-Host "  $publicUrl"
    Write-Host ""
    Write-Host "On iPhone Safari: open the URL above, enter token: $token"
    Write-Host "Share -> Add to Home Screen for a full-screen app icon."
    Write-Host ""
    Write-Host "Note: quick tunnel URL changes each time you restart this script."
    Write-Host "For a stable URL, install Tailscale (see SETUP-REMOTE.md)."
} else {
    Write-Host "Tunnel starting - check $tunnelLog for the public URL." -ForegroundColor Yellow
}

try {
    while ($true) {
        if ($serverJob.State -eq "Failed") {
            Write-Host "Server stopped unexpectedly:" -ForegroundColor Red
            Receive-Job $serverJob
            break
        }
        Start-Sleep -Seconds 5
    }
} finally {
    Write-Host ""
    Write-Host "Stopping..."
    Stop-Job $tunnelJob -ErrorAction SilentlyContinue
    Remove-Job $tunnelJob -ErrorAction SilentlyContinue
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue
    Stop-PortListener -Port $port
}
