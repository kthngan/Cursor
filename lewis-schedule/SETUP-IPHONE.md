# iPhone test setup (Windows PC + same Wi-Fi)

Follow these steps on your **Windows PC** where the Cursor repo lives, then open the app on your **iPhone**.

## 1. Pull the latest code

In Cursor or PowerShell, from your repo root (`C:\Users\user\Documents\Cursor`):

```powershell
git pull origin cursor/lewis-schedule-web-d552
```

Or merge [PR #1](https://github.com/kthngan/Cursor/pull/1) into `main` first, then `git pull`.

## 2. Create your `.env` file

```powershell
cd lewis-schedule\agent
copy .env.example .env
notepad .env
```

Set these values:

| Variable | Value |
|----------|--------|
| `ACCESS_TOKEN` | `lulufeijai` (or your own password) |
| `HOST` | `0.0.0.0` |
| `WORKSPACE_DIR` | `C:\Users\user\Documents\Cursor` |
| `CURSOR_API_KEY` | Your key from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations) — **only needed for screenshot import** |

Grid editing and export work **without** `CURSOR_API_KEY`.

## 3. Start the server

```powershell
cd C:\Users\user\Documents\Cursor\lewis-schedule
.\start.ps1
```

Leave this window open. You should see:

```
Open http://127.0.0.1:8790
iPhone (same Wi-Fi): http://192.168.x.x:8790
Access token: lulufeijai
```

Note the `192.168.x.x` address — that is what you use on your iPhone.

### If Windows Firewall blocks the phone

When prompted, allow **Python** on **Private networks**. Or run once as admin:

```powershell
New-NetFirewallRule -DisplayName "Lewis Schedule 8790" -Direction Inbound -Protocol TCP -LocalPort 8790 -Action Allow
```

## 4. Open on your iPhone

1. Connect iPhone to the **same Wi-Fi** as the PC.
2. Safari → `http://192.168.x.x:8790` (use the IP from step 3).
3. Access token: **`lulufeijai`**
4. Tap **Connect**.

The confirmed schedule for **week of 14 Jul 2026** loads automatically.

## 5. Add to Home Screen (optional)

Safari → **Share** (square with arrow) → **Add to Home Screen** → Add.

Opens full-screen like an app — no App Store.

## 6. Try these features

| Action | How |
|--------|-----|
| View this week | Already loaded (Mon 14 Jul – Sun 20 Jul) |
| Edit a slot | Tap activity or caregiver text |
| Swap days | Tap one row, then another row |
| Export to WhatsApp | **Export** → **Copy** or **Share** → WhatsApp |
| Reload confirmed week | **Reload this week** |
| Screenshot import | **Import screenshot** (needs `CURSOR_API_KEY`) |

## Confirmed schedule loaded

| Day | Morning | Afternoon |
|-----|---------|-----------|
| Mon | Nancy class 9am — Por por | Regular day — Por por |
| Tue | Little Habs Interview 8:45am — Mah mah | Regular day — Mah mah |
| Wed | Regular day — Por por | Regular day — Por por |
| Thu | Nancy class 9am — Por por | Regular day — Mah mah |
| Fri | Judy class — Por por | Mah mah (PM); Por por overnight — Mah mah |
| Sat | Regular day — Por por | Regular day — Por por |
| Sun | Regular day — Por por | Regular day — Por por |

Friday **night** (Por por sleepover) is noted in the Friday PM activity line.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| iPhone cannot connect | Same Wi-Fi? Correct IP? Firewall rule? PC server still running? |
| Invalid token | Must match `ACCESS_TOKEN` in `.env` exactly |
| Old schedule showing | Safari → clear site data, or tap **Reload this week** |
| Import screenshot fails | Set `CURSOR_API_KEY` in `.env` and restart server |

## Away from home (optional later)

Use **Tailscale** on PC + iPhone, then open `http://<pc-tailscale-name>:8790` from anywhere.
