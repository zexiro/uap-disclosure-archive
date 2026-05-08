# Deploy guide

## Live deployment

- **URL:** https://uap-mirror-production.up.railway.app
- **Railway project:** [`uap-disclosure-archive`](https://railway.com/project/77a51944-0f75-4cad-bfd7-56bad406a8eb)
- **Service:** `uap-mirror` (single service, single container)
- **Volume:** `/app/raw` — persistent, holds the 5.6 GB of bulk media
- **Region:** europe-west4-drams3a

## Day-to-day workflow

```bash
# Edit code locally, then:
git add ...
git commit -m "your change"
git push
```

If the **GitHub Actions deploy** is wired (one-time setup below), `git push`
to `main` triggers a Railway redeploy automatically. The volume persists, so
the data survives.

If GH Actions isn't wired yet, run:

```bash
railway up --service uap-mirror --detach -m "your release note"
```

## One-time GitHub Actions setup

Enable hands-off deploys on every push:

1. **Generate a Railway token:**

   ```bash
   # Visit https://railway.com/account/tokens
   # Create a token scoped to project: uap-disclosure-archive
   # Copy the token value
   ```

2. **Add it to the GitHub repo:**

   ```bash
   gh secret set RAILWAY_TOKEN -b "<paste-token-here>" -R zexiro/uap-disclosure-archive
   ```

3. **Done.** `.github/workflows/deploy.yml` runs on every push to `main` and
   calls `railway up`. Inspect runs at:
   <https://github.com/zexiro/uap-disclosure-archive/actions>

## Initial cold-start

When Railway boots a fresh container with an empty volume, the entrypoint:

1. Starts the HTTP server immediately (search UI works from the bundled
   `ui/search-index.json`).
2. Kicks off `scripts/refresh.sh` in the background:
   - re-fetches `uap-csv.csv` from war.gov,
   - downloads all PDFs / images / videos to the volume (~5–10 min on
     Railway's bandwidth),
   - OCRs PDFs (parallel, ~10–20 min total for the heavy FBI scans),
   - rebuilds links + search index + vault.

After the first run, the volume is warm. Subsequent boots skip the pipeline
entirely (the cron only re-runs it when war.gov publishes a new tranche).

## Custom domain (optional)

```bash
# In Railway dashboard → uap-mirror → Settings → Domains → Custom Domain
# Add your domain, then add the CNAME they give you to your DNS.
```

## Useful commands

```bash
railway logs   --service uap-mirror --lines 200
railway logs   --service uap-mirror --build --lines 200
railway status --service uap-mirror --json
railway variable list --service uap-mirror --json
railway redeploy --service uap-mirror --yes      # rebuild from same source
railway restart  --service uap-mirror --yes      # just restart the process
```

## When new disclosure tranches drop

Nothing to do — the cron checks `war.gov/UFO/uap-csv.csv` every 6 hours,
hashes it, and only re-runs the pipeline when the hash changes. New
relationship links computed across the combined corpus mean cross-tranche
patterns surface automatically.

If you want to force an immediate refresh:

```bash
railway run --service uap-mirror "bash scripts/refresh.sh"
```
