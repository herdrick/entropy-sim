# Entropy Simulator (Web Version)

A browser-based interactive simulator for exploring information theory. It samples from a hidden source of single-field numerical data, builds a live histogram / probability distribution, and plots how that distribution/model's entropy and per-event surprisal change over time.

## Running it

This is a static site with no build step. Serve the `web/` directory with any HTTP server. (It cannot be opened directly as a `file://` URL because ES module imports require HTTP.)

```bash
# Python (from the repo root or from web/)
python -m http.server 8080 --directory web/
```
Then open http://localhost:8080

## No dependencies to install

All libraries (Three.js, Chart.js, jStat) are loaded from CDN at runtime.

## What you'll see

Four panels update in real time:

| Panel | Contents |
|---|---|
| Live Histogram | 3D bar chart of observed sample counts (20 bins over [0, 1]) |
| Entropy Over Time | Model entropy in bits as more events arrive |
| Surprisal Stream | Per-event surprisal (dots) and running average |
| Latest Event | Value, surprisal, and cumulative event count for the most recent sample |

## Controls

- **Play / Pause** — start or stop automatic sampling
- **Speed slider** — 1–1000 events per second
- **Reset** — clear all data and stop playback
- **Reveal Distribution** — overlay the source's PDF on the histogram and show the source entropy as a reference line on the entropy chart
- **Mystery Source** — switch between six sources (Uniform, several Beta distributions, and a mixture)
- **Add Event** — type a value in [0, 1] and press Enter or click Add to inject it manually

