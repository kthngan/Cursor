# Enable GitHub Pages (required once)

You saw **“There isn’t a GitHub Pages site here”** because Pages is **not turned on** yet.  
The app files are already in this repo — you only need to flip one switch.

---

## Option A — Easiest (recommended)

Works on iPhone Safari.

1. Open **https://github.com/kthngan/Cursor/settings/pages**
2. Sign in as **kthngan** if asked.
3. Under **Build and deployment** → **Source**, choose **Deploy from a branch**.
4. Set:
   - **Branch:** `main`
   - **Folder:** `/docs`
5. Tap **Save**.
6. Wait **1–2 minutes** (refresh the Pages settings page — you should see a green URL).

**Open on iPhone:** **https://kthngan.github.io/Cursor/**

---

## Option B — GitHub Actions (automatic deploys)

Use this if you prefer deploys to run on every push to `main`.

1. Open **https://github.com/kthngan/Cursor/settings/pages**
2. Under **Build and deployment** → **Source**, choose **GitHub Actions** (not “Deploy from a branch”).
3. Go to **https://github.com/kthngan/Cursor/actions**
4. Open **Deploy Lewis Schedule to GitHub Pages** → **Run workflow** → **Run workflow**.
5. When the run finishes (green tick), open **https://kthngan.github.io/Cursor/**

---

## Until Pages is enabled — temporary link

Safari → tap **Open the page** on the warning screen:

**https://raw.githack.com/kthngan/Cursor/main/docs/index.html**

This works without GitHub Pages but is less convenient than Option A.

---

## Checklist

| Step | Done? |
|------|-------|
| Repo has `docs/index.html` | Yes (already committed) |
| You enabled Pages in Settings | **You must do this** |
| Open `https://kthngan.github.io/Cursor/` | After enable |

---

## Still not working?

| Symptom | Fix |
|---------|-----|
| “There isn’t a GitHub Pages site here” | Pages not enabled — complete Option A above |
| 404 after enabling | Wait 2–5 min; hard-refresh Safari |
| Settings page won’t load | Make sure you’re logged in as **kthngan** |
| No `/docs` folder option | Pull latest `main` — folder is at repo root |

---

## Add to iPhone Home Screen

Safari → **https://kthngan.github.io/Cursor/** → **Share** → **Add to Home Screen**
