# Entropy Simulator (Web Version)

Made to explore changes in surprisals and entropy as the distribution (under which the surprisals are calculated) changes.

Samples an event from a source distribution, calculates that event's surprisal under the current probability distribution (model), and updates the probability distribution to include this new event.

Uses Three.js, Chart.js, and jStat, loaded from a CDN at runtime.

## Running it

```bash
python -m http.server 8080
```
