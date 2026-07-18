# Bootstrap: clone or update kthngan/Cursor on this Windows PC, then start Lewis Schedule.
# Run once (no repo needed):
#   irm https://raw.githubusercontent.com/kthngan/Cursor/main/lewis-schedule/clone-to-pc.ps1 | iex
# Or save this file and: powershell -ExecutionPolicy Bypass -File clone-to-pc.ps1

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/kthngan/Cursor.git"
$DefaultTarget = Join-Path $env:USERPROFILE "Documents\Cursor"

param(
    [string]$Target = $DefaultTarget,
    [switch]$SkipStart
)

function Write-Step($msg) {
    Write-Host ""
    Write-Host ">> $msg" -ForegroundColor Cyan
}

function Ensure-Git {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        return (Get-Command git).Source
    }

    $portable = Join-Path $Target ".tools\mingit\cmd\git.exe"
    if (Test-Path $portable) {
        return $portable
    }

    return $null
}

function Clone-FromZip {
    param([string]$Dest)
    Write-Step "No git found — downloading repo as ZIP"
    $parent = Split-Path -Parent $Dest
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $zip = Join-Path $env:TEMP "Cursor-main.zip"
    $extract = Join-Path $env:TEMP "Cursor-main"
    Invoke-WebRequest -Uri "https://github.com/kthngan/Cursor/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path $extract) {
        Remove-Item $extract -Recurse -Force
    }
    Expand-Archive -Path $zip -DestinationPath $env:TEMP -Force
    if (Test-Path $Dest) {
        Write-Host "Folder exists: $Dest — updating files from ZIP"
        Copy-Item -Path (Join-Path $extract "*") -Destination $Dest -Recurse -Force
    } else {
        Move-Item -Path $extract -Destination $Dest
    }
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Clone Cursor repo to this PC ===" -ForegroundColor Green
Write-Host "Target: $Target"

$parentDir = Split-Path -Parent $Target
if (-not (Test-Path $parentDir)) {
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
}

$git = Ensure-Git

if ($git -and (Test-Path (Join-Path $Target ".git"))) {
    Write-Step "Repo already cloned — pulling latest"
    Set-Location $Target
    & $git pull origin main
} elseif ($git) {
    if (Test-Path $Target) {
        Write-Step "Folder exists but is not a git repo — cloning into it"
        Set-Location (Split-Path -Parent $Target)
        & $git clone $RepoUrl (Split-Path -Leaf $Target)
    } else {
        Write-Step "Cloning $RepoUrl"
        Set-Location $parentDir
        & $git clone $RepoUrl (Split-Path -Leaf $Target)
    }
} else {
    Clone-FromZip -Dest $Target
    $git = Ensure-Git
}

if (-not (Test-Path $Target)) {
    throw "Clone failed — folder not found: $Target"
}

Write-Step "Done. Repo is at: $Target"
Write-Host ""
Write-Host "Next:" -ForegroundColor Yellow
Write-Host "  cd $Target\lewis-schedule"
Write-Host "  .\run-all.ps1"
Write-Host ""
Write-Host "Browser: http://127.0.0.1:8790  |  Token: lewis-2026-test"
Write-Host "iPhone (same Wi‑Fi): http://<your-PC-IP>:8790"
Write-Host ""

if (-not $SkipStart) {
    $runAll = Join-Path $Target "lewis-schedule\run-all.ps1"
    if (Test-Path $runAll) {
        $answer = Read-Host "Start Lewis Schedule now? (Y/n)"
        if ($answer -eq "" -or $answer -match '^[Yy]') {
            Set-Location (Split-Path $runAll)
            & powershell -ExecutionPolicy Bypass -File $runAll
        }
    }
}
