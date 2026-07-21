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
pip install -r requirements.txt && playwright install chromium
```

Do **not** use `playwright install --with-deps chromium` on Render's native
(non-Docker) runtime — `--with-deps` tries to `apt-get install` system
libraries via `sudo`/`su` to root, and the Render build user isn't root and
has no passwordless sudo, so the build fails with `su: Authentication
failure`. Plain `playwright install chromium` only downloads the browser
binary (no system package install, no root needed) and builds fine.

If Render's base image is missing an OS-level shared library Chromium needs
at runtime (e.g. `libnss3`, `libgbm1`), the browser will simply fail to
launch — `fetch_html_rendered()` wraps that in a try/except and fails soft
(returns `""`), so the app keeps running exactly as it did before this
feature; you'd just never see `"JS-rendered"` in the diagnostics. The real
fix in that case is switching the service to a Docker-based Render runtime
so the base image (and its apt-installed deps) is under your control.

If the install step is skipped entirely, the app still runs fine too — the
fallback just no-ops (`_PLAYWRIGHT_OK` stays `False`) and behavior is
identical to before this was added. It can also be disabled at any time
without a redeploy by setting the environment variable `ENABLE_JS_RENDER=0`.

Other env vars for this fallback (all optional):
- `JS_RENDER_MAX_CONCURRENT` (default `2`) — max pages rendering at once,
  across all EANs being processed concurrently. Lower this if the instance is
  memory-constrained.
- `JS_RENDER_TIMEOUT_MS` (default `15000`) — per-page render timeout.
