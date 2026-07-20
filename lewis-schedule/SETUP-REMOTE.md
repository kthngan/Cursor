# Remote access (not on same Wi-Fi)

The local server listens on port **8790**. Use one of these options from your iPhone anywhere.

## Option A — Cloudflare quick tunnel (easiest, already set up)

From `lewis-schedule`:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-remote.ps1
```

This starts the server **and** a public HTTPS URL like `https://xxxx.trycloudflare.com`.

- Open that URL on iPhone Safari
- Access token: **`lulufeijai`** (or whatever you set in `agent\.env`)
- **Screenshot import works** (needs `CURSOR_API_KEY` in `.env`)
- Leave the PowerShell window open while using the app
- The URL **changes each restart** — run the script again and use the new link

After a Sunday PC restart: run `start-remote.ps1` again.

---

## Option B — Tailscale (stable URL, recommended long-term)

Best if you want the same address every time without copying a new link.

1. Install **Tailscale** on PC and iPhone: https://tailscale.com/download
2. Sign in with the same account on both devices
3. On PC, run:

```powershell
cd C:\Users\user\Documents\Cursor\lewis-schedule
.\start.ps1
```

4. On iPhone Safari, open:

```
http://<your-pc-tailscale-name>:8790
```

Find your PC name in the Tailscale app (e.g. `desktop-abc123`). Works on cellular, other Wi-Fi, abroad.

---

## Option C — Same Wi-Fi only (no tunnel)

```powershell
.\start.ps1
```

Use `http://192.168.x.x:8790` from iPhone on the **same** network.

---

## Security

- Always keep **`ACCESS_TOKEN`** set in `agent\.env` — the internet tunnel is public if someone has the URL
- Do not share the tunnel URL publicly
- Add **`CURSOR_API_KEY`** only on your PC (never commit `.env`)

---

## Auto-start after reboot (optional)

Task Scheduler → Create task:

- Trigger: **At log on**
- Action: `powershell.exe -ExecutionPolicy Bypass -File C:\Users\user\Documents\Cursor\lewis-schedule\start-remote.ps1`
- Run whether user is logged on or not (if you want it always available)

For Tailscale + `start.ps1` instead, use that script path in the action.
