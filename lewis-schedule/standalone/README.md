# Lewis Schedule — Standalone (offline)

Single-file web app for iPhone: no server, no auth, works fully offline after first load.

## URLs

| Purpose | URL |
|---------|-----|
| **GitHub Pages** (after deploy) | https://kthngan.github.io/Cursor/ |
| **HTML preview** (test before Pages) | https://htmlpreview.github.io/?https://raw.githubusercontent.com/kthngan/Cursor/cursor/lewis-schedule-web-d552/lewis-schedule/standalone/index.html |

## Deploy to GitHub Pages

1. Push `lewis-schedule/standalone/index.html` to the repo.
2. In GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Use a workflow that publishes the standalone file (or the repo root) to Pages.
4. After deploy, open the **GitHub Pages** URL above on your iPhone.

## iPhone: Add to Home Screen

Safari → **Share** → **Add to Home Screen**. Opens full-screen like a native app.

## What works offline (this file)

- View the confirmed week (14 Jul 2026 template auto-loads on first visit)
- Edit activity and caregiver text on any slot
- Swap slots: tap-tap on phone, drag-drop on desktop
- Previous / next week navigation
- **Reload this week** — resets to embedded confirmed template
- **Export** — copy or share text for WhatsApp
- State persists in `localStorage` between visits

## What needs the server

| Feature | Why |
|---------|-----|
| **Import screenshot** | Requires the Python agent + Cursor API on your PC (see `lewis-schedule/SETUP-IPHONE.md`) |
| **Live template updates** | Server serves `/api/template`; offline build embeds a fixed JSON snapshot |
| **Access token / multi-user** | Server-only auth; standalone has no gate |

When you are home on the same Wi-Fi as your PC, use the full app at `http://<pc-ip>:8790` for screenshot import. The offline banner and Import button explain this in the app.

## Files

| File | Role |
|------|------|
| `index.html` | Self-contained app (CSS, JS, template JSON inline) |
| `README.md` | This file |

Source of truth for the template: `lewis-schedule/templates/default_template.json` (embedded as `CONFIRMED_TEMPLATE` in `index.html`).
