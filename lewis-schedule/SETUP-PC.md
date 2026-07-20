# Set up on your Windows PC

The cloud agent runs on a remote machine and **cannot** run commands on your PC. Use one of the options below on **your** computer.

## Option A — One command (easiest)

Open **PowerShell** and paste:

```powershell
irm https://raw.githubusercontent.com/kthngan/Cursor/main/lewis-schedule/clone-to-pc.ps1 | iex
```

This clones (or updates) the repo to:

`C:\Users\<you>\Documents\Cursor`

Then offers to run `run-all.ps1` to start the local server.

## Option B — Git clone manually

If you already have [Git for Windows](https://git-scm.com/download/win):

```powershell
cd $env:USERPROFILE\Documents
git clone https://github.com/kthngan/Cursor.git
cd Cursor\lewis-schedule
powershell -ExecutionPolicy Bypass -File .\run-all.ps1
```

If the folder already exists:

```powershell
cd $env:USERPROFILE\Documents\Cursor
git pull origin main
cd lewis-schedule
powershell -ExecutionPolicy Bypass -File .\run-all.ps1
```

## Option C — Download ZIP (no git)

1. Open https://github.com/kthngan/Cursor/archive/refs/heads/main.zip
2. Extract to `Documents\Cursor` (rename `Cursor-main` → `Cursor` if needed)
3. Run:

```powershell
cd $env:USERPROFILE\Documents\Cursor\lewis-schedule
powershell -ExecutionPolicy Bypass -File .\run-all.ps1
```

## After setup

| What | Value |
|------|--------|
| Local URL | http://127.0.0.1:8790 |
| Access token | `lulufeijai` |
| Same Wi-Fi | `.\start.ps1` → open `http://192.168.x.x:8790` on iPhone |
| **Internet (any network)** | `.\start-remote.ps1` → use the `https://….trycloudflare.com` URL |
| Screenshot import | Add `CURSOR_API_KEY` to `lewis-schedule\agent\.env` |
| iPhone (offline, no import) | https://kthngan.github.io/Cursor/v6.html |

See **[SETUP-REMOTE.md](./SETUP-REMOTE.md)** for Tailscale (stable URL) and auto-start after reboot.

## Troubleshooting

- **`git` not recognized** — use Option A (ZIP fallback) or install Git for Windows.
- **Wrong folder** — your path may be `C:\Users\kthng\Documents\Cursor` instead of `user`; the script uses your Windows username automatically.
- **Port in use** — change `PORT` in `lewis-schedule\agent\.env`.
