# Moving the NRL Model to GitHub + Cloud Automation

Result: the weekly automation runs on GitHub's servers (laptop can be off),
all ledger files live in the repo and update automatically, and the dashboard
is viewable from any device via Streamlit Community Cloud. Cost: $0.

## Step 1 - Prepare the folder (5 min)

1. In your `nrl-model` folder, create a folder named `.github`, inside it a
   folder named `workflows`, and move `nrl_workflow.yml` there, renamed to
   `nrl.yml`. Final path:  `.github/workflows/nrl.yml`
   (In Terminal:  `mkdir -p .github/workflows && mv nrl_workflow.yml .github/workflows/nrl.yml`)
2. Make sure `requirements.txt` is in the folder.
3. DELETE `odds_api_key.txt` from the folder before uploading - the key must
   never be committed. (The workflow recreates it at runtime from a secret.)
4. Optional: remove the local Mac schedule so runs don't happen twice:
   `bash setup_automation.sh remove`

## Step 2 - Create the GitHub repo (10 min)

Easiest path for a non-git-user - GitHub Desktop:
1. Create a free account at github.com, install GitHub Desktop.
2. GitHub Desktop -> File -> Add Local Repository -> choose the nrl-model
   folder (it will offer to "create a repository" - accept, defaults fine).
3. Commit all files ("Initial commit"), then Publish repository.
   IMPORTANT: tick "Keep this code private".
   (The 54MB player_stats.jsonl is under GitHub's 100MB limit - fine.)

## Step 3 - Add the API key secret (2 min)

On github.com -> your repo -> Settings -> Secrets and variables -> Actions
-> New repository secret:
  Name:  ODDS_API_KEY
  Value: (paste the key from your odds_api_key.txt)

## Step 4 - Verify (2 min)

Repo -> Actions tab -> "NRL Model Automation" -> Run workflow (manual button).
Watch it go green (~10 min). Afterwards the repo's CSV files will show a
fresh commit from "nrl-model-bot". From then on it runs on schedule
automatically.

## Step 5 - Host the dashboard (10 min)

1. Go to share.streamlit.io, sign in with GitHub.
2. New app -> pick your repo -> main branch -> file: dashboard.py -> Deploy.
3. You'll get a private URL like https://yourname-nrl.streamlit.app that
   works from any device and always shows the latest committed data.
   (For a private repo, keep the app access restricted to your account in
   the app settings.)

## Ongoing

- Your laptop copy: open GitHub Desktop and click "Pull" any time you want
  the latest ledgers locally. If you edit files locally, commit + push -
  but avoid editing CSVs that the bot also writes, or you'll get conflicts.
- Manual SGM combo prices: easiest via github.com - open sgm_ledger.csv in
  the repo, click the pencil icon, edit, commit. Or edit locally and push.
- Timezone note: GitHub cron is UTC and does not follow daylight saving.
  When Sydney shifts to AEDT in October (finals), runs land one hour
  earlier local time - harmless for this schedule.
- If a run fails, GitHub emails you. The log is in the Actions tab.
