# Entropy Simulator (Web Version)

Made to explore changes in surprisals and entropy as the distribution (under which the surprisals are calculated) changes.

Samples an event from a source distribution, calculates that event's surprisal under the current probability distribution (model), and updates the probability distribution to include this new event.

Uses Three.js, Chart.js, jStat, loaded from a CDN at runtime.

- `1-var/`  Single random variable version.
- `2-var/`  Two random variable version.

## Running it

```bash
python -m http.server 8080
```
