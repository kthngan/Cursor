$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$EnvFile = Join-Path $Root ".env"
$ExampleFile = Join-Path $Root ".env.example"
$Cloudflared = Join-Path $Root "cloudflared.exe"
$Port = 8787

if (-not (Test-Path $EnvFile)) {
    Copy-Item $ExampleFile $EnvFile
    Write-Host ""
    Write-Host "Created .env — edit it before starting:" -ForegroundColor Yellow
    Write-Host "  1. Set CURSOR_API_KEY from https://cursor.com/dashboard/integrations"
    Write-Host "  2. Set ACCESS_TOKEN to a password you'll use on your phone"
    Write-Host "  3. Confirm WORKSPACE_DIR points to your local folder"
    Write-Host ""
    notepad $EnvFile
    exit 1
}

Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

if (-not $env:CURSOR_API_KEY) {
    Write-Host "CURSOR_API_KEY is missing in .env" -ForegroundColor Red
    exit 1
}
if (-not $env:ACCESS_TOKEN) {
    Write-Host "ACCESS_TOKEN is missing in .env" -ForegroundColor Red
    exit 1
}
if ($env:PORT) { $Port = [int]$env:PORT }

if (-not (Test-Path $Cloudflared)) {
    Write-Host "Downloading cloudflared..."
    Invoke-WebRequest `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $Cloudflared
}

Write-Host ""
Write-Host "Starting local agent on http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "Workspace: $($env:WORKSPACE_DIR)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Keep this window open. Your phone URL will appear below." -ForegroundColor Yellow
Write-Host ""

$serverProcess = Start-Process `
    -FilePath "python" `
    -ArgumentList "server.py" `
    -WorkingDirectory $Root `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 4

try {
    & $Cloudflared tunnel --url "http://127.0.0.1:$Port"
} finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
    }
}
