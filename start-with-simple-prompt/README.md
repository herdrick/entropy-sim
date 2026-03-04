# Interactive Entropy Simulator

An interactive desktop visualizer that teaches entropy and surprisal by sampling from hidden probability distributions in real time. Watch a live histogram grow, observe how entropy converges, and try to guess each mystery source before revealing it.

---

## What it does

The simulator streams random samples from one of six "mystery sources" (probability distributions). As samples arrive, four panels update live:

- **Live Histogram** — the empirical distribution of samples so far
- **Entropy Over Time** — how the estimated Shannon entropy (in bits) evolves as more data arrives
- **Surprisal Stream** — the surprisal of each individual event, color-coded and with a running average
- **Latest Event** — a number-line marker showing where the most recent sample landed, plus its surprisal and the current entropy estimate

You can switch sources, reset, control sample speed, and reveal the true distribution and theoretical entropy at any time.

---

## Prerequisites

- Python 3.8 or later
- A desktop environment capable of opening a GUI window (the app uses the `TkAgg` matplotlib backend, which requires Tk to be installed)

**Tk availability by platform:**

| Platform | Likely status |
|----------|---------------|
| macOS (Homebrew Python) | Usually included; install `python-tk` via Homebrew if missing |
| Ubuntu/Debian | Install with `sudo apt install python3-tk` |
| Windows | Included with the standard Python installer from python.org |

---

## Installation

### 1. Navigate to this directory

```bash
cd start-with-simple-prompt
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` lists three packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, histogram binning, probability calculations |
| `scipy` | Statistical distributions (uniform, beta, mixture) and numerical integration |
| `matplotlib` | Interactive GUI window, all four plot panels, buttons, sliders, radio buttons |

> Note: The `requirements.txt` also contains a comment showing an equivalent conda install command (`conda install --file requirements.txt`) if you prefer a conda environment.

No API keys or external services are required.

---

## Usage

```bash
python entropy_sim.py
```

A 14x9 inch dark-themed window opens. The simulation starts paused.

### Controls

| Control | Description |
|---------|-------------|
| **Play / Pause** button | Start or stop the sample stream |
| **Reset** button | Clear all data and restart from zero |
| **Reveal Distribution** button | Toggle the true PDF overlay on the histogram and show the theoretical entropy line |
| **Speed slider** | Set samples per second (1 to 100) |
| **Mystery Source radio buttons** | Switch between Source A through Source F |

### Sources

The six mystery sources are:

| Label | True distribution |
|-------|------------------|
| Source A | Uniform(0, 1) — maximum entropy, flat histogram |
| Source B | Beta(2, 5) — skewed toward low values |
| Source C | Beta(0.5, 0.5) — U-shaped, peaks near 0 and 1 |
| Source D | Beta(5, 5) — bell-shaped, concentrated near 0.5 |
| Source E | Beta(0.3, 0.3) — strongly U-shaped, very concentrated at the edges |
| Source F | Mixture: 0.5 * Beta(2,2) + 0.5 * Beta(20,20) — bimodal |

---

## What to expect

1. Press **Play**. Bars in the histogram will begin filling in and the entropy trace will start climbing toward the source's theoretical entropy.
2. Watch the **Surprisal Stream**: rare events (out in the tails) appear red and high on the plot; common events appear blue and low.
3. The **running average surprisal** (yellow line) converges toward the theoretical entropy — this is the empirical demonstration of the Shannon source coding theorem.
4. Once the histogram shape looks stable, click **Reveal Distribution** to overlay the true PDF (yellow curve) and see the theoretical entropy dashed line in the entropy panel.
5. Use the **radio buttons** to switch sources and compare how entropy and surprisal differ across distributions.
