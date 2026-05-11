# Deploy guide

## Live deployment

- **URL:** https://uap-mirror-production.up.railway.app
- **Railway project:** [`uap-disclosure-archive`](https://railway.com/project/77a51944-0f75-4cad-bfd7-56bad406a8eb)
- **Service:** `uap-mirror` (single service, single container)
- **Volume:** `/app/raw` — persistent, holds the 5.6 GB of bulk media
- **Region:** europe-west4-drams3a

## How deploys actually work

There are **two paths**. Only one is automatic.

### 1. Railway's native GitHub integration (primary, automatic)

The Railway project is connected to this GitHub repo at the project level
(configured once in the Railway dashboard, not in this codebase). Every push
to `main` triggers a webhook that rebuilds and redeploys the service with no
intervention from CI. The persistent volume survives, so the corpus data is
preserved across deploys.

```bash
# Edit code locally, then:
git add ...
git commit -m "your change"
git push                              # Railway picks it up via webhook
```

To verify it's still wired, check the deploy log in the Railway dashboard
under `uap-disclosure-archive → uap-mirror → Deployments`.

### 2. `.github/workflows/deploy.yml` (manual fallback)

A `workflow_dispatch`-only workflow lets you force a redeploy from the GitHub
UI even when no commit has been pushed (useful after rotating env vars or
invalidating a cache). It does **not** run on every push — Railway native
already handles that, and racing two deploys against each other causes
flapping.

Trigger it via the Actions tab → `Deploy to Railway (manual fallback)` →
`Run workflow`, or:

```bash
gh workflow run deploy.yml
```

Requires the `RAILWAY_TOKEN` repo secret (see setup below). The workflow
fails fast with a clear error if the secret is missing.

### Direct CLI (for emergencies)

If GitHub is offline or you need to deploy from a non-`main` branch:

> **Note:** The server now runs via uvicorn + starlette (ASGI) instead of the
> previous `python scripts/serve.py` (ThreadingHTTPServer). The entrypoint is
> unchanged from Railway's perspective — `entrypoint.sh` still starts the
> foreground process. Locally you can run:
> ```bash
> uvicorn scripts.serve:app --host 0.0.0.0 --port 8000 --workers 1
> ```

```bash
railway up --service uap-mirror --detach -m "your release note"
```

## One-time setup for the manual workflow

Only needed if you want the `workflow_dispatch` fallback to work.

1. **Generate a Railway token** at <https://railway.com/account/tokens>,
   scoped to project `uap-disclosure-archive`. Copy the token value.

2. **Add it to the GitHub repo:**

   ```bash
   gh secret set RAILWAY_TOKEN -b "<paste-token-here>" -R zexiro/uap-disclosure-archive
   ```

3. **Trigger the workflow once** to verify it works:

   ```bash
   gh workflow run deploy.yml
   gh run watch
   ```

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

## Post-deploy smoke checks

Run these after every deploy that touches `scripts/serve.py`, `entrypoint.sh`, or `requirements.txt`. Each one tests a route class that has bitten us before:

```bash
HOST=https://uapdisclosuremirror.com

# 1. Server is up at all
curl -fsS -o /dev/null -w "healthz %{http_code}\n"           "$HOST/healthz"

# 2. Front door — directory index.
#    REGRESSION HISTORY: the ASGI port (commit 651d6fa) lost stdlib
#    SimpleHTTPRequestHandler's directory-index behaviour. /ui/ matched the
#    /ui/{path:path} route with empty path → resolved to the ui/ directory
#    → is_file() False → 404. Fixed in ccd4127. Always test /ui/ explicitly,
#    not just /ui/index.html — they hit different code paths.
curl -fsS -o /dev/null -w "/ %{http_code}\n"                 "$HOST/"
curl -fsS -o /dev/null -w "/ui/ %{http_code}\n"              "$HOST/ui/"
curl -fsS -o /dev/null -w "/ui/index.html %{http_code}\n"    "$HOST/ui/index.html"

# 3. Static asset serving
curl -fsS -o /dev/null -w "/raw/records.json %{http_code}\n" "$HOST/raw/records.json"

# 4. RAG endpoint (sources event must arrive even with a dummy key)
curl -fsS -N -m 6 -X POST "$HOST/api/ask" \
  -H 'content-type: application/json' \
  -d '{"question":"smoke"}' | head -3

# 5. WebSocket upgrade handshake (must return 101 Switching Protocols).
#    Note: a plain HTTP GET on /ws/collab returns 404 in Starlette — that
#    is normal (there's no separate "wrong protocol" handler), so we have
#    to send the actual upgrade headers to know the route is wired.
curl -fsS -o /dev/null -w "/ws/collab upgrade %{http_code} (expect 101)\n" \
  --http1.1 \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "$HOST/ws/collab"
```

Expected statuses: `200` for everything HTTP, `302` for `/`, `101` for the WebSocket upgrade. Any other code means something regressed — check `railway logs --latest --lines 200` and compare against the previous deploy.

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
