# Entropy & Surprisal Explorer

An interactive browser-based tool for building probability distributions from scratch and watching entropy change in real time. You generate random events, bin them however you like, and the app computes and displays the Shannon entropy of the resulting distribution.

## What It Does

The app opens in your browser as a Bokeh server application with two linked plots:

- **Rug plot** — shows individual sampled events as vertical tick marks along the x-axis.
- **P distribution** — a bar chart of the probability mass in each bin, with the current Shannon entropy (in bits) displayed in the title.

You interact with it through four controls:

| Control | What it does |
|---|---|
| **Add events** | Samples `n` values from a uniform distribution and appends them to the rug plot. |
| **n =** (text input) | Sets how many events to generate (default: 1000). |
| **Make distribution from events** | Bins the current events and redraws the probability bar chart. |
| **Clear events** | Wipes the rug plot; leaves the distribution unchanged. |
| **Divide a bin** | Reveals a text input where you type a bin edge value (e.g. `0.5`) and press Enter. This splits the bin that contains that value, creating a new bin edge. The distribution is immediately recomputed. |

The outermost bins always extend to negative and positive infinity. When you pan or zoom the chart, those bins visually stretch to fill the viewport, making it clear they are unbounded.

**Laplace smoothing** is used: each bin gets a pseudocount added to its count of events. This, combined with the outermost bins extending to negative and positive infinity, prevents P of zero for any possible event.

## Prerequisites

- Python 3.8 or newer
- If you wish to run the tests, you will also need Node.js and npm.

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
```

This installs:

- **numpy** — array math used throughout the app
- **scipy** — generates uniform random samples for the event stream
- **bokeh** — the interactive plotting framework and server that powers the UI

No API keys or external accounts are required.

## Running the App

```bash
bokeh serve foo.py
```

Bokeh will print a URL to the terminal, typically:

```
http://localhost:5006/foo
```

## What to Expect

When the page loads you will see:

1. An empty rug plot at the top.
2. A single bar spanning the full viewport (one infinite bin) with an entropy of 0.0000 bits in the chart title.

A typical workflow:

1. Click **Add events** to generate 1000 random samples (uniform between 0 and 1). The rug plot fills with tick marks.
2. Click **Make distribution from events** to bin those events and update the bar chart. Entropy will increase from 0.
3. Click **Divide a bin**, type `0.5`, and press Enter. The single bar splits at 0.5 into two bins and the entropy updates.
4. Keep adding bin edges to subdivide the distribution further. Each split that creates more-equal bins raises entropy; splits that isolate very few events may raise or lower it depending on the data.
5. Click **Add events** again to accumulate more data and re-run **Make distribution from events** to see how a larger sample affects the shape.

Pan and zoom with your mouse — the outermost bars will always stretch to fill the visible range.

## Running the Tests

The test suite uses Playwright (a browser automation framework) to verify that the infinite bin edges correctly track the viewport on pan and zoom.

### Install Node dependencies

```bash
npm install
```

### Install the Playwright browser

```bash
npx playwright install chromium
```

### Run the tests

```bash
npx playwright test
```

The test runner starts the Bokeh server automatically on port 5006, runs five browser-based tests in `tests/infinite_bins.spec.js`, and shuts the server down when finished.

If the Bokeh server is already running on port 5006, Playwright will reuse it instead of starting a new one.

## Project Structure

```
chained-surprisal-distributions/
├── foo.py                      # Main Bokeh application
├── events.py                   # Event-generation helper (uniform sampler)
├── requirements.txt            # Python dependencies
├── package.json                # Node dependencies (Playwright)
├── playwright.config.js        # Playwright configuration
└── tests/
    └── infinite_bins.spec.js   # Browser tests for infinite-edge behaviour
```

## Troubleshooting

**`bokeh: command not found`**
Bokeh was not installed or the virtual environment is not active. Run `pip install -r requirements.txt` inside the activated environment.

**Port 5006 is already in use**
Either stop the existing process or pass a different port: `bokeh serve foo.py --port 5007`. If you change the port, update `playwright.config.js` to match before running tests.

**The browser shows a blank page or "connection refused"**
The Bokeh server may still be starting up. Wait a moment and refresh. If it persists, check the terminal output for errors.

**Playwright tests fail with "browserType.launch: Executable doesn't exist"**
Run `npx playwright install chromium` to download the required browser binary.

**Events are generated but the bar chart does not change**
You need to click **Make distribution from events** after adding events. Adding events only updates the rug plot; the distribution is recomputed on demand.
