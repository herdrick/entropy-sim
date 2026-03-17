# Interactive Entropy Simulator

A desktop app that visualizes entropy and surprisal by sampling from a probability distribution in real time.

## What it does

Streams random samples from a source distribution. **The probability distribution is updated with every new event seen** — as samples arrive, the empirical distribution evolves and four panels update live:

- **Live Histogram** — empirical distribution of samples so far (scroll to zoom)
- **Entropy Over Time** — Shannon entropy estimate (bits) as the distribution is refined by incoming data
- **Surprisal Stream** — per-event surprisal, color-coded, with running average converging toward the source entropy
- **Latest Event** — most recent sample on a number line with its surprisal

Controls: play/pause, reset, speed slider (1-1000 samples/sec), and manual event entry.

## Desktop version (Python/Matplotlib)

### Dependencies
The `requirements.txt` lists three packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, histogram binning, probability calculations |
| `scipy` | Statistical distributions (uniform, beta, mixture) and numerical integration |
| `matplotlib` | Interactive GUI window, all four plot panels, buttons, sliders, radio buttons |

### Setup

Requires Python 3.8+ with Tk support (included on most systems).

```bash
cd new-distribution-with-each-event
pip install -r requirements.txt
python entropy_sim.py
```

## Web version

There is also a browser-based version with no build step or Python dependencies. See [web/README.md](web/README.md) for setup and usage.
