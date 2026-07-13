---
name: run-chained-surprisal-distributions
description: Start the chained-surprisal-distributions Bokeh app and check it in a browser. Use when asked to run this app, take a screenshot of it, or verify a code change works.
---

Check if it's already running first (the user often keeps one up under `entr` for auto-reload):

```bash
curl -sf http://localhost:5006/continuous_main >/dev/null && echo "already running"
```

If not, start it on a fresh random port:

```bash
cd chained-surprisal-distributions
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bokeh-1
PORT=$((5100 + RANDOM % 900))
nohup bokeh serve main.py fixed_point.py continuous_main.py continuous_fixed_point.py --port $PORT \
  > /tmp/bokeh.stdout.txt 2> /tmp/bokeh.stderr.txt &
echo $! > /tmp/bokeh.pid
timeout 20 bash -c "until curl -sf http://localhost:$PORT/continuous_main >/dev/null; do sleep 0.5; done"
```

Then use Playwright to hit `http://localhost:$PORT/continuous_main` (or `/main`, `/fixed_point`,
`/continuous_fixed_point`) and click around to check your change. If using the `playwright` MCP
server and you hit "Browser is already in use", add `--isolated` to its args in `~/.claude.json`
so concurrent sessions don't share one on-disk browser profile.

Stop the server after with `kill $(cat /tmp/bokeh.pid)`.

## Gotchas

- `test_app.py` and `README.md` reference a `foo.py` app and a `tests/` dir that no longer exist
  in this repo — ignore them, there's no working automated test suite here.
- Bokeh renders into `.bk-Column`, not `.bk-root`.
