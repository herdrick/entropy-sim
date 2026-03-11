# Interactive Entropy Simulator

A desktop app that visualizes entropy and surprisal by sampling from probability distributions in real time.

## What it does

Streams random samples from one of six source distributions. As samples arrive, four panels update live:

- **Live Histogram** — empirical distribution of samples so far (scroll to zoom)
- **Entropy Over Time** — Shannon entropy estimate (bits) as more data arrives
- **Surprisal Stream** — per-event surprisal, color-coded, with running average converging toward the source entropy
- **Latest Event** — most recent sample on a number line with its surprisal

Controls: play/pause, reset, speed slider (1-1000 samples/sec), source selector, reveal source distribution, and manual event entry.

## Dependencies
The `requirements.txt` lists three packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, histogram binning, probability calculations |
| `scipy` | Statistical distributions (uniform, beta, mixture) and numerical integration |
| `matplotlib` | Interactive GUI window, all four plot panels, buttons, sliders, radio buttons |

## Setup

Requires Python 3.8+ with Tk support (included on most systems).

```bash
cd start-with-simple-prompt
pip install -r requirements.txt
python entropy_sim.py
```
