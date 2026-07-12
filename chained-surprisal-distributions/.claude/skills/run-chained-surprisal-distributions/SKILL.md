---
name: run-chained-surprisal-distributions
description: Build, run, and drive the chained-surprisal-distributions Bokeh app (main.py, fixed_point.py, continuous_main.py, continuous_fixed_point.py). Use when asked to start this app, take a screenshot of it, click its buttons, or verify a UI change actually works.
---

This is a browser-based Bokeh server app (Python) with four entry points —
`main.py`, `fixed_point.py`, `continuous_main.py` (most actively developed —
recent commits all touch this one), `continuous_fixed_point.py` — served
together by one `bokeh serve` process. Drive it with
`.claude/skills/run-chained-surprisal-distributions/driver.mjs`, a small
Playwright REPL that launches its own isolated Chromium (won't conflict with
any shared `playwright-mcp` browser another session is using).

All paths below are relative to `chained-surprisal-distributions/`.

## Prerequisites

Python deps live in the `bokeh-1` conda env (already created on this
machine):

```bash
~/miniconda3/envs/bokeh-1/bin/python -c "import bokeh, numpy, scipy" # bokeh 3.8.2, numpy 2.4.2, scipy 1.17.0
```

Node deps for the driver:

```bash
npm install   # installs playwright (already in package.json)
npx playwright install chromium   # only if not already cached
```

## Run (agent path)

**Check first whether a dev server is already running** — the user
normally keeps one up under `entr` for auto-reload (see Gotchas):

```bash
curl -sf http://localhost:5006/continuous_main >/dev/null && echo "already running"
```

If nothing is running, start one (any free port; use 5006 if free):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bokeh-1
nohup bokeh serve main.py fixed_point.py continuous_main.py continuous_fixed_point.py --port 5006 \
  > /tmp/bokeh.stdout.txt 2> /tmp/bokeh.stderr.txt &
echo $! > /tmp/bokeh.pid
timeout 20 bash -c 'until curl -sf http://localhost:5006/continuous_main >/dev/null; do sleep 0.5; done'
```

Stop it later with `kill $(cat /tmp/bokeh.pid)`.

Then drive it — pipe commands to the driver's stdin:

```bash
node .claude/skills/run-chained-surprisal-distributions/driver.mjs <<'EOF'
nav http://localhost:5006/continuous_main
click Add events
screenshot /tmp/after_add.png
click View derived distribution
screenshot /tmp/derived.png
console
EOF
```

Commands the driver understands:

| command | what it does |
|---|---|
| `nav <url>` | navigate, wait for the Bokeh root (`.bk-Column`) to render |
| `click <exact button text>` | click a button by its accessible name |
| `fill <label prefix> <text>` | fill a text input by placeholder prefix |
| `press <key>` | e.g. `press Enter` (needed for "Divide a bin" / "Add event at value…") |
| `wait <ms>` | fixed pause — only for animation/step playback, not for loading |
| `screenshot <path>` | full-page PNG |
| `console` | prints any captured console errors so far |

Verified working flow (`continuous_main`, http://localhost:5006/continuous_main):
`nav` → `click Add events` → `click View derived distribution` renders the
KDE-derived P1 plot with an entropy readout in the title (e.g. "entropy =
0.2916 bits") and zero console errors.

For `main.py` / `fixed_point.py` / `continuous_fixed_point.py`, same
pattern — just change the path in `nav` (`/main`, `/fixed_point`,
`/continuous_fixed_point`).

## Run (human path)

```bash
conda activate bokeh-1
bokeh serve main.py fixed_point.py continuous_main.py continuous_fixed_point.py
```

Opens on http://localhost:5006/ — each file is a separate app path (e.g.
`/continuous_main`). Ctrl-C to stop.

## Test

`test_app.py` and `README.md` are stale: both reference a `foo.py` app and
a `tests/` directory that no longer exist in this repo — don't run them,
they'll fail on a missing file, not on your change. There is currently no
working automated test suite; use the driver above to verify changes.

## Gotchas

- **The user often already has a server running** under `entr` for
  auto-reload: `find . | grep \.py$ | entr -r bokeh serve main.py
  fixed_point.py continuous_main.py continuous_fixed_point.py`. If you
  edit any served `.py` file, entr restarts the server immediately —
  don't check logs or grep for new PIDs to confirm the restart, just
  `nav` again. If the page looks wrong, that's a real bug, not a loading
  delay.
- **`.bk-root` is not the right selector** to wait on (some docs/examples
  elsewhere assume it) — this Bokeh version renders into `.bk-Column`.
- **`ProtocolError("Token is expired...")` in stderr** after any restart
  is just a stale browser tab trying to reconnect to the old session —
  ignore it.
- **Multiple `bokeh serve` files → one process, multiple paths.** Don't
  start four separate `bokeh serve` calls; pass all four files to one
  invocation and each becomes its own `/<name>` path.
- **Don't reuse a shared `playwright-mcp` Chromium profile** for
  scripted checks — if another session holds
  `~/Library/Caches/ms-playwright-mcp/mcp-chrome-*`, `browser_navigate`
  fails with "Browser is already in use". The driver here launches its
  own throwaway Chromium instance instead, so it never hits that lock.
- **`bokeh serve main.py` warns** "It looks like you might be running
  the main.py of a directory app directly" — harmless in this project's
  layout (these are standalone single-file apps, not a directory app);
  ignore it.

## Troubleshooting

- **`page.waitForSelector('.bk-Column')` times out**: the server isn't
  actually up yet, or you navigated to a path with no matching file
  (e.g. `/foo` — doesn't exist). Confirm with `curl -sf
  http://localhost:5006/continuous_main`.
- **`click <text>` errors with no matching element**: button text must
  match exactly, including the "(one by one)" suffix where present —
  check the visible label with a `screenshot` first.
