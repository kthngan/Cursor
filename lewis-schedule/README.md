# Lewis Schedule (web)

Private weekly schedule for Lewis — half-day slots (AM/PM), drag-and-drop editing, Composer-powered screenshot import, and WhatsApp-friendly export.

## Layout in this repo

```
lewis-schedule/
├── agent/          FastAPI backend (Composer 2.5 via cursor-sdk)
├── web/            Browser UI (Add to Home Screen on iPhone)
└── templates/      default_template.json

.cursor/skills/lewis-schedule-import/   Composer skill (repo root)
```

## iPhone test run

**GitHub Pages must be enabled once** — see **[ENABLE-PAGES.md](./ENABLE-PAGES.md)** if you see *“There isn’t a GitHub Pages site here”*.

After enable: **https://kthngan.github.io/Cursor/**

Temporary (no setup): **https://raw.githack.com/kthngan/Cursor/main/docs/index.html**

See also **[OFFLINE-IPHONE.md](./OFFLINE-IPHONE.md)** and **[SETUP-IPHONE.md](./SETUP-IPHONE.md)**.

## Quick start

1. Copy env file:

```bash
cd lewis-schedule/agent
cp .env.example .env
```

2. Set in `.env`:
   - `CURSOR_API_KEY` — from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations)
   - `ACCESS_TOKEN` — password for the web app
   - `WORKSPACE_DIR` — path to this repo root (so Composer loads the skill)

3. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

4. Open `http://127.0.0.1:8790` on your phone or desktop.

### iPhone

- Safari → your URL (via Tailscale or tunnel) → **Share → Add to Home Screen**
- No App Store required

### Remote access

Expose port `8790` with Tailscale, Cloudflare Tunnel, or similar. Keep `ACCESS_TOKEN` set.

## Features

| Feature | Notes |
|---------|--------|
| Weekly grid | 14 half-day slots; prev/next week |
| Template | Load `templates/default_template.json` |
| Drag-and-drop | Drag rows on desktop; tap-tap swap on phone |
| Screenshot import | Composer 2.5 reads partial updates, asks questions, returns a patch |
| Export | Copy or Share sheet text for WhatsApp |

## Composer import

Screenshots often show **one activity on one day**. The agent skill at `.cursor/skills/lewis-schedule-import/SKILL.md` instructs Composer to ask clarifying questions and return JSON patches — not replace the whole week.

**Image uploads:** use JPEG/PNG. If the backend runs on Linux, test image import early (SDK bridge can be sensitive to large images). Resize screenshots on the phone if needed.

## Customise the template

Edit `templates/default_template.json` (activities, caregivers, default week pattern).

Schedule edits in the browser are stored in **localStorage** on each device.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Auth check |
| GET | `/api/template` | Default template JSON |
| POST | `/schedule/import/start` | Screenshot + schedule → Composer |
| POST | `/schedule/import/continue` | Answer follow-up questions |
| DELETE | `/schedule/import/{thread_id}` | End import session |
