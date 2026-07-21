# parcels-search

## Deployment: Render build command

This app uses a headless-browser (Playwright/Chromium) fallback for product
pages that verify but come back data-thin from a plain HTTP fetch (JS-rendered
"drive"/click-and-collect storefronts — see `fetch_html_rendered` in
`food_pipeline.py`). `pip install -r requirements.txt` only installs the
Playwright Python client, **not** the Chromium binary — that needs an extra
install step.

**Update the Render service's Build Command to:**

```
pip install -r requirements.txt && playwright install --with-deps chromium
```

If this step is skipped, the app still runs fine — the fallback just no-ops
(`_PLAYWRIGHT_OK` stays `False`) and behavior is identical to before this was
added. It can also be disabled at any time without a redeploy by setting the
environment variable `ENABLE_JS_RENDER=0`.

Other env vars for this fallback (all optional):
- `JS_RENDER_MAX_CONCURRENT` (default `2`) — max pages rendering at once,
  across all EANs being processed concurrently. Lower this if the instance is
  memory-constrained.
- `JS_RENDER_TIMEOUT_MS` (default `15000`) — per-page render timeout.
