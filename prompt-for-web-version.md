# Prompt: Interactive Entropy Simulator — Web Version

Build a static web application (no backend) that is an interactive entropy and surprisal simulator. It streams random samples from hidden probability distributions and visualizes how entropy converges in real time. This is a rewrite of an existing Python/matplotlib desktop app for the web.

## Architecture

- **No build step.** Single `index.html` that loads JS and CSS. Use ES modules.
- **CDN dependencies:**
  - **Three.js** (with OrbitControls) — for the histogram panel only
  - **Chart.js** — for the other three chart panels (entropy line, surprisal scatter, latest event)
  - **jStat** (`jstat`) — for Beta distribution math (PDF, CDF, random sampling)
- **No framework.** Vanilla JS, CSS Grid layout.

## Layout

- Dark theme (background `#1a1a2e`, panel backgrounds `#16213e`, accent `#e94560`, highlight `#e2d810`).
- **2x2 grid of chart panels** taking up most of the viewport, with a **controls sidebar** on the right.
- Responsive — should work reasonably on different screen sizes.

## The Four Panels

### Panel 1: Live Histogram (top-left) — THREE.JS

This is the key panel. Render it with **Three.js using an orthographic camera**.

- **1D mode (default):** Bins defined by `bin_edges` (initially 21 edges = 20 bins over [0, 1]). The leftmost and rightmost bins are open intervals extending to -/+infinity (see Binning section). Render as 3D bars (box geometries) viewed from directly above/front in orthographic projection so it looks like a flat 2D bar chart. The z-axis is degenerate — bars have minimal depth. If the user rotates the camera (via OrbitControls), they see it's a flat chart — this is fine and expected.
- Bar heights = probability (count / total). Update bar mesh heights each time an event arrives.
- Show bin edge markers at 0 and 1.
- The leftmost and rightmost bars are displayed at the same width as other bars by default, but visually stretch to fill the visible x-range when the user zooms out past [0, 1].
- **Reveal mode:** When the user clicks "Reveal Distribution", overlay the true PDF curve (scaled by bin width to match histogram units). Use a line geometry or a ribbon mesh in yellow (`#e2d810`).
- Axis labels: "Value" (x), "Probability" (y). Render axis labels and tick marks as HTML overlays positioned over the canvas, or as CSS2DRenderer labels.
- Scroll-to-zoom on the x-axis, centered on cursor position.

**Future 2D extension (design for but don't build yet):** The architecture should make it natural to later switch to 20x20 bins where bars rise in the y-axis and the camera starts top-down. Rotating reveals the 3D bar chart. Keep this in mind for how you structure the histogram mesh generation — e.g., store bars in a grid-addressable structure, not a flat array tied to 1D assumptions.

### Panel 2: Entropy Over Time (top-right) — CHART.JS

- Line chart. X-axis = event count, Y-axis = entropy in bits.
- Single line (`#e94560`) showing the **model entropy** (from `compute_model_entropy()`) after each event.
- When revealed: dashed horizontal line (`#e2d810`) at the **source entropy** (from `compute_source_entropy()`), with a legend label like "Source entropy: X.XXX bits".
- Auto-scale axes.

### Panel 3: Surprisal Stream (bottom-left) — CHART.JS

- Scatter chart. X-axis = event index, Y-axis = surprisal in bits.
- Each event is a dot (uniform color, no color-coding — just use `#e94560` or similar).
- Overlay a **running average line** in yellow (`#e2d810`), which converges toward the theoretical entropy (demonstrating the Shannon source coding theorem).
- Auto-scale axes.

### Panel 4: Latest Event (bottom-right) — CHART.JS or HTML

- Shows a number line from 0 to 1 with tick marks every 0.1.
- A large marker dot showing where the latest sample landed.
- Text displaying:
  - Value: X.XXXX
  - Surprisal: X.XX bits
  - Entropy of model: X.XXXXXXX bits
  - Events: N
- This can be a simple canvas/Chart.js panel or even just styled HTML — whatever is cleanest.

## Controls Sidebar

- **Play / Pause** button — starts/stops the sample stream
- **Reset** button — clears all data, resets charts
- **Reveal Distribution** button — toggles the true PDF overlay on histogram + theoretical entropy line on entropy chart. When revealed, button text changes to the distribution name (e.g. "Beta(2,5)").
- **Speed slider** — samples per second, range 1 to 1000, default 10
- **Mystery Source** radio buttons — Source A through Source F
- **Manually add event** button + text input — lets user type a numeric value and add it as an event

## Mystery Sources (Probability Distributions)

```
Source A: Uniform(0, 1)
Source B: Beta(2, 5)
Source C: Beta(0.5, 0.5)
Source D: Beta(5, 5)
Source E: Beta(0.3, 0.3)
Source F: Mixture — 0.5 * Beta(2, 2) + 0.5 * Beta(20, 20)
```

Use jStat for Beta PDF, CDF, and random sampling. For Source F (mixture), sample by flipping a coin then drawing from the selected component. Compute its PDF/CDF as the weighted sum.

## Entropy & Surprisal Math

All entropy and surprisal values are in **bits** (log base 2).

### Binning
- The fundamental data structure is `bin_edges` — an array of edges. Initially `linspace(0, 1, 21)` (21 edges, producing 20 bins). The bin count is always just `bin_edges.length - 1`. We will make `bin_edges` user-configurable in a future version.
- The leftmost bin is **open on the left**: (-inf, bin_edges[1]). It catches any value below the first interior edge, including values below 0.
- The rightmost bin is **open on the right**: [bin_edges[n-1], +inf) where n = bin_edges.length - 1. It catches any value at or above the last interior edge, including values above 1.
- The interior bins are half-open intervals: [bin_edges[i], bin_edges[i+1]).
- There are no special "overflow" bins — the first and last bins simply extend to infinity. All bins are treated uniformly in the entropy/surprisal math.

### Bin index mapping
```
n_bins = bin_edges.length - 1

if value < bin_edges[1]:           bin = 0              (leftmost open bin)
if value >= bin_edges[n_bins - 1]: bin = n_bins - 1     (rightmost open bin)
otherwise:                         bin = searchsorted(bin_edges, value) - 1, clamped to [0, n_bins - 1]
```

### Model entropy (Laplace-smoothed)
This is the entropy of the model built from observed data. Called `compute_model_entropy()`.
```
smoothed_counts = counts + 1          (add 1 to ALL bins)
total = sum(smoothed_counts)
probs = smoothed_counts / total
model_entropy = -sum(probs * log2(probs))
```

### Surprisal of an event
```
bin = get_bin_index(value)
smoothed_counts = counts + 1
total = sum(smoothed_counts)
prob = smoothed_counts[bin] / total
surprisal = -log2(prob)
```
**Important:** Compute surprisal BEFORE adding the event to the histogram counts. This gives the surprisal of the event given the model *before* it saw that event.

### Theoretical entropy of source
This is the true binned entropy of the underlying distribution. Called `compute_source_entropy()`. It is what the model entropy converges toward.
```
cdf_at_edges = source_cdf(bin_edges)           // CDF evaluated at all edges
bin_probs = diff(cdf_at_edges)                 // probabilities from consecutive CDF differences
// Adjust endpoints for open bins:
bin_probs[0] += cdf_at_edges[0]                // add P(X < 0) to leftmost bin
bin_probs[last] += 1.0 - cdf_at_edges[last+1] // add P(X >= 1) to rightmost bin
source_entropy = -sum(p * log2(p) for p in bin_probs if p > 0)
```

## Timing / Animation

- Use `setInterval` for the sample stream. Interval = `1000 / speed` ms, clamped to a minimum of 10ms.
- On each tick (if playing): sample from current source, compute surprisal (before updating counts), update counts, compute entropy, update all four panels.
- Chart.js updates: call `chart.update('none')` (skip animations) for performance during streaming.
- Three.js updates: update bar mesh heights and call `renderer.render(scene, camera)`.

## File Structure

```
web/
  index.html      — main page, includes CSS inline or in <style>, loads app.js
  app.js          — all application logic
```

Keep it to these two files. Inline the CSS in index.html.

## Key Behavioral Details

- Switching sources triggers a reset (clears all data).
- The reveal toggle is independent of play/pause — you can reveal while paused or playing.
- The running average surprisal line uses the cumulative mean of all surprisal values so far (not a windowed average).
- The speed slider should update the timer interval immediately when dragged.
- Histogram y-axis shows probability (count/total), not raw counts.
- When no events have occurred yet, show empty charts with reasonable default axis ranges.
