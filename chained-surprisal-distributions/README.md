# Entropy & Surprisal Explorer

An interactive browser-based toolkit for building probability distributions from sampled events and exploring entropy, surprisal, and self-information "chains" in real time. It ships as **four separate Bokeh apps** — a discrete (histogram) version and a continuous (KDE) version, each with a plain explorer and a "fixed point" variant that repeatedly re-derives the surprisal distribution of its own surprisal distribution until it stops changing.

## What It Does

All four apps share the same core idea: you generate random events from a chosen distribution family, bin (or fit a density to) those events into a distribution **P**, and the app shows you the Shannon entropy of that distribution. You can then **derive** a child distribution — the surprisal (`-log2(probability)`) of each event under its parent — and repeat, building a chain of distributions.

### `iterated_surprisal_distributions.py` — discrete explorer (`/iterated_surprisal_distributions`)

- **Rug plot / transport controls** — step through accumulated events one at a time or all at once, via a history slider, back/forward buttons, and an "Add events (one by one)" animated playback button.
- **Event generation** — pick a distribution family (Uniform, Normal, Beta, Exponential) from a dropdown, tune its parameters with sliders, and choose to **Append** new events or **Replace** all events. You can also inject a single event at an exact value.
- **P distribution bar chart** — a histogram over bin edges you control, with Shannon entropy (in bits) shown in the plot title.
- **Bin edge controls** — a "Split point" slider plus "Freeze edge" to lock it in, an "Add single edge" text box, and "evenly spaced" left/right/count sliders to generate many edges at once. The outermost bins always extend to ±infinity and visually stretch to fill the viewport on pan/zoom.
- **Laplace smoothing** — a Gaussian(μ, σ) prior is blended into each bin's count via a "Prior strength α" slider, so no bin's probability is ever exactly zero.
- **Probability vs. probability density** toggle, and fixed vs. adaptive Y-axis scaling.
- **Derive chains** — click "View derived distribution" to create a child node whose events are the surprisal values of the parent's events; each child can itself be split into bins and further derived, building an arbitrarily long chain/tree. A "Control all descendants' parameters" checkbox gangs a node's bin/prior settings down to its descendants.
- **KL divergence and Wasserstein (W1) distance** are displayed between each node and its parent/child.
- **Hover-to-trace** — hovering over a bin highlights the corresponding bins in every ancestor and descendant node, showing how a value's surprisal propagates through the chain.
- **1/2/3-column layout** toggle for arranging multiple derived nodes.

### `find_fixed_point.py` — discrete fixed-point explorer (`/find_fixed_point`)

A single P1 node plus its first surprisal distribution S(P1), with the same bin-edge and prior controls as `iterated_surprisal_distributions.py`. Instead of manually deriving children, it automatically **iterates** the surprisal transform (re-bin → re-derive → re-bin …) up to 1000 times per event batch and reports how many iterations it took to converge (tolerance adjustable via a slider), or that it didn't converge. It also:

- Tracks session-best convergence records.
- Accumulates every converged fixed-point probability vector and visualizes them across four extra panels: a 3D simplex plot, a radial/spoke plot, a scatter-plot matrix, and a parallel-coordinates plot (all restricted to the most frequently active bins, with an optional "lock" to a fixed set of bins).
- Overlays converged fixed-point distributions on a scatter chart with adjustable opacity.

### `continuous_iterated_surprisal_distributions.py` — continuous (KDE) explorer (`/continuous_iterated_surprisal_distributions`)

The same chaining/deriving idea as `iterated_surprisal_distributions.py`, but each node fits a **continuous density** to its events instead of binning them — via Gaussian KDE, adaptive KDE, a Gaussian mixture model, or a B-spline fit (selectable per node), blended with a Gaussian(μ, σ) prior. Deriving a child computes `S(x) = -log2(density(x) · Δx)` at each event, where Δx is an adjustable "bin width" stand-in. Additional controls include a bandwidth/smoothing slider, number-of-GMM-components slider, an event rug overlay with adjustable opacity, and a "trace new events" mode that highlights where newly added events land. A "Working…" indicator appears during slower recomputations (e.g. propagating settings to many descendants). The four bin-simplex viz panels from `find_fixed_point.py` have no continuous analogue and are not present here.

### `continuous_find_fixed_point.py` — continuous fixed-point explorer (`/continuous_find_fixed_point`)

The continuous analogue of `find_fixed_point.py`: a P1 node and its S(P1) node, both density-fit rather than binned, iterated to a fixed point the same way. Includes a progression view and an overlay chart of the density's shape across iterations, plus the same rug/trace overlays as `continuous_iterated_surprisal_distributions.py`.

## Prerequisites

- Python 3.8 or newer
- **matplotlib** is required by `find_fixed_point.py` (via `viz_simplex3d.py`, for the 3D simplex panel) but is **not listed in `requirements.txt`** — install it separately (`pip install matplotlib`) if you plan to run `find_fixed_point.py`.
- If you wish to run `test_app.py`, you will also need a browser installed via Playwright (see Running the Tests below).

## Installation

### 1. Navigate to the directory

```bash
cd chained-surprisal-distributions
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install matplotlib   # needed by find_fixed_point.py; not in requirements.txt
```

`requirements.txt` installs:

- **numpy** — array math used throughout every app
- **scipy** — random-sample generation (`events.py`) and, in the continuous apps, KDE/spline density fitting
- **bokeh** — the interactive plotting framework and server that powers the UI
- **pytest** / **pytest-playwright** — used by `test_app.py`

No API keys or external accounts are required.

## Running the App

The four apps are served together as one Bokeh multi-app server. Run it under `entr` so the server automatically restarts whenever any `.py` file in the directory changes:

```bash
find . | grep \.py$ | entr -r bokeh serve iterated_surprisal_distributions.py find_fixed_point.py continuous_iterated_surprisal_distributions.py continuous_find_fixed_point.py
```

`find . | grep \.py$` lists every `.py` file in the directory, and pipes that list into `entr`, which watches those files and reruns the given command each time one of them changes. The `-r` flag tells `entr` to restart the command (rather than just rerun it) on each change, which is what you want for a long-running server — it kills the old `bokeh serve` process and starts a fresh one, so edits to any app's code take effect without you having to stop and restart the server by hand.

Bokeh will print four URLs to the terminal:

```
http://localhost:5006/iterated_surprisal_distributions
http://localhost:5006/find_fixed_point
http://localhost:5006/continuous_iterated_surprisal_distributions
http://localhost:5006/continuous_find_fixed_point
```

Open whichever one you want to explore — they run independently in the same server process.

## What to Expect

When `/iterated_surprisal_distributions` or `/continuous_iterated_surprisal_distributions` loads you will see:

1. An empty rug/event area at the top and a single P distribution (one infinite bin, entropy 0.0000 bits).
2. Below that, event-generation controls (distribution family, parameters, Append/Replace, n=, Add events).

A typical workflow on `/iterated_surprisal_distributions`:

1. Pick a family (e.g. Normal) and click **Add events** to generate a batch of samples.
2. Click **View derived distribution** to spawn a P node, then drag the **Split point** slider and click **Freeze edge** to add bin edges — the bar chart and entropy update immediately.
3. Click **View derived distribution** again on that node to create a surprisal child, and repeat to build a chain. Watch the KL/W1 numbers between adjacent nodes.
4. Hover over a bin to see the "trace" highlight flow up/down the chain.

On `/find_fixed_point` or `/continuous_find_fixed_point`, just add events — the app automatically iterates and reports the number of iterations to convergence (or "did not converge") without any manual deriving.

## Running the Tests

`test_app.py` is a `pytest` + Playwright end-to-end suite (it launches its own Bokeh server subprocess and drives it with a real browser). **Note:** as of this writing it still references the app's old filename — it starts `bokeh serve foo.py` and requests `http://localhost:5007/foo` — so it will fail to launch until those references are updated to `iterated_surprisal_distributions.py` / `/iterated_surprisal_distributions`. There is no `tests/` directory or Playwright JS spec file in this project despite `playwright.config.js` and `package.json` existing; those currently point at the same stale `foo.py` name.

Once the filename references are fixed, the intended usage is:

```bash
pip install pytest-playwright
playwright install chromium
pytest test_app.py            # headless
pytest test_app.py --headed   # watch in a real browser
```

## Project Structure

```
chained-surprisal-distributions/
├── iterated_surprisal_distributions.py            # Discrete explorer app (/iterated_surprisal_distributions)
├── find_fixed_point.py                            # Discrete fixed-point app (/find_fixed_point)
├── continuous_iterated_surprisal_distributions.py # Continuous (KDE) explorer app (/continuous_iterated_surprisal_distributions)
├── continuous_find_fixed_point.py                 # Continuous (KDE) fixed-point app (/continuous_find_fixed_point)
├── events.py                                      # Event-generation helper (distribution families/samplers)
├── bin_selection.py                               # Bin-frequency tracker + "lock bins" UI (used by find_fixed_point.py)
├── compare_fixed_points.py                        # Standalone script: do different starting distributions
│                                                   #   converge to the same fixed point? (not a Bokeh app)
├── viz_simplex3d.py                               # 3D simplex viz panel (find_fixed_point.py only; needs matplotlib)
├── viz_radial.py                                  # Radial/spoke viz panel (find_fixed_point.py only)
├── viz_scatter_matrix.py                          # Scatter-plot-matrix viz panel (find_fixed_point.py only)
├── viz_parallel_coords.py                         # Parallel-coordinates viz panel (find_fixed_point.py only)
├── test_app.py                                    # Pytest + Playwright end-to-end tests (see caveat above)
├── requirements.txt                               # Python dependencies
├── package.json                                   # Node dependency (Playwright), stale test config
└── playwright.config.js                           # Playwright config, stale (still points at foo.py)
```

## Troubleshooting

**`bokeh: command not found`**
Bokeh was not installed or the virtual environment is not active. Run `pip install -r requirements.txt` inside the activated environment.

**`ModuleNotFoundError: No module named 'matplotlib'` when loading `find_fixed_point.py`**
`matplotlib` is required by `viz_simplex3d.py` but isn't in `requirements.txt`. Run `pip install matplotlib`.

**Port 5006 is already in use**
Either stop the existing process or pass a different port: `bokeh serve iterated_surprisal_distributions.py find_fixed_point.py continuous_iterated_surprisal_distributions.py continuous_find_fixed_point.py --port 5007`.

**The browser shows a blank page or "connection refused"**
The Bokeh server may still be starting up. Wait a moment and refresh. If it persists, check the terminal output for errors.

**`pytest test_app.py` fails immediately trying to launch `foo.py`**
The test file hasn't been updated since the app was renamed from `foo.py` to `iterated_surprisal_distributions.py`. Update the `bokeh serve` command and URL inside `test_app.py` (and `playwright.config.js`, if you use it) to reference `iterated_surprisal_distributions.py` / `/iterated_surprisal_distributions` before running.

**Events are generated but the distribution doesn't change**
Adding events updates the rug/history state; if a node exists, its distribution recomputes automatically on **Add events**. If nothing has been derived yet, click **View derived distribution** first to create the initial P node.

**`ProtocolError("Token is expired...")` in server logs**
Harmless — it comes from stale browser tabs trying to reconnect to an old session after the server restarts (e.g. via `entr`). Safe to ignore.
