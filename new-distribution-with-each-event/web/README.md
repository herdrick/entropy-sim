# Entropy Simulator (Web Version)

Made to explore changes in surprisals and entropy as the distribution (under which the surprisals are calculated) changes. 

Samples an event from a source distribution, calculates that event's surprisal under the current probability distribution (model), and updates the probability distribution to include this new event.

Uses Three.js, Chart.js, jStat, loaded from a CDN at runtime.

## Running it

Static web app, no build step. Serve the `web/` directory with any HTTP server. (It cannot be opened directly as a `file://` URL because ES module imports require HTTP.)

```bash
# Python (from the repo root or from web/)
python -m http.server 8080 --directory .
```
Then open http://localhost:8080
