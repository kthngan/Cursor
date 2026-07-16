# Project Memory

## Git / GitHub

- This repo is tracked with Git. Remote: `https://github.com/kthngan/Cursor.git`, branch `main`.
- **Do not install or reinstall Git.** Use the portable MinGit already in the workspace:
  - `C:\Users\user\Documents\Cursor\.tools\mingit\cmd\git.exe`
  - In PowerShell, prefix commands with `& ".tools\mingit\cmd\git.exe"` from the repo root.
- System `git` is not on PATH in Cursor shells; that is expected.
- `.tools/` is gitignored (portable Git stays local only).
- GitHub push has worked before via Git Credential Manager; complete any browser sign-in prompt if push hangs.

## What to commit

- Source scripts (`.py`, `.ps1`, requirements, cursor rules).
- Do **not** commit: `Data/`, `Reports/`, `.env`, credentials, generated outputs, `.tools/`.

## Commit author (optional)

If git identity is unset, use env vars for one commit (do not run `git config`):

```powershell
$env:GIT_AUTHOR_NAME = "Cursor Agent"
$env:GIT_AUTHOR_EMAIL = "cursor-agent@local"
$env:GIT_COMMITTER_NAME = "Cursor Agent"
$env:GIT_COMMITTER_EMAIL = "cursor-agent@local"
```

## HTML reports

- Default output: `C:\Users\user\Documents\Cursor\Reports` (see `.cursor/rules/report-output-conventions.mdc`).
