# Interactive Entropy Simulator

Visualizes entropy and surprisal by pulling a stream of observations from a hidden random source and updating the observed probability distribution with each new event.

## What it does

Streams random samples from a source distribution. **The probability distribution is updated with every new event seen** — as samples arrive, the empirical distribution evolves and four panels update live:

- **Live Histogram** — empirical distribution of samples so far (scroll to zoom)
- **Entropy Over Time** — Shannon entropy estimate (bits) as the distribution is refined by incoming data
- **Surprisal Stream** — per-event surprisal, color-coded, with running average converging toward the source entropy
- **Latest Event** — most recent sample on a number line with its surprisal

**Archived.** Superseded by the web version at `../../new-distribution-with-each-event/`.

## Desktop version (Python/Matplotlib)

### Dependencies
See `requirements.txt`.

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
