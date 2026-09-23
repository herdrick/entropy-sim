# Entropy & Surprisal Explorer

This was made to explore the idea that the surprisal of an event is itself an event. The probability distribution of those surprisal events can itself be used as another model under which you can calculate surprisals of events. So, you can chain together such distributions and pass batches of events into the top of that chain, and get distributions of surprisals based on distributions of surprisals, and so on. You can play with that directly with iterated_surprisal_distributions.py.  

Invariably a fixed point is found. (Why is obvious enough when you think about it.) You can explore that with find_fixed_point.py.

There are continuous-space version of those two, but the results you get are just quirks in whatever density-fitting technique you use.

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
pip install matplotlib   # needed by find_fixed_point.py; not in requirements.txt
```

## Running the App

The four apps are served together as one Bokeh multi-app server. 

```bash
bokeh serve iterated_surprisal_distributions.py find_fixed_point.py continuous_iterated_surprisal_distributions.py continuous_find_fixed_point.py
```

If you are changing the code, you may find it convenient to run it under `entr` so the server automatically restarts whenever any relevant file in the directory changes, ex.:

```bash
find . | grep \.py$ | entr -r bokeh serve iterated_surprisal_distributions.py find_fixed_point.py continuous_iterated_surprisal_distributions.py continuous_find_fixed_point.py
```

## Running the Tests

`test_app.py` is a `pytest` + Playwright end-to-end suite (it launches its own Bokeh server subprocess on port 5007 and drives it with a real browser).

```bash
pip install pytest-playwright
playwright install chromium
pytest test_app.py            # headless
pytest test_app.py --headed   # watch in a real browser
```

## Project Structure

```
chained-surprisal-distributions/
├── iterated_surprisal_distributions.py            # Discrete explorer app (/iterated_surprisal_distributions)
├── find_fixed_point.py                            # Discrete fixed-point app (/find_fixed_point)
├── continuous_iterated_surprisal_distributions.py # Continuous (KDE) explorer app (/continuous_iterated_surprisal_distributions)
├── continuous_find_fixed_point.py                 # Continuous (KDE) fixed-point app (/continuous_find_fixed_point)
├── events.py                                      # Event-generation helper (distribution families/samplers)
├── bin_selection.py                               # Bin-frequency tracker + "lock bins" UI (used by find_fixed_point.py)
├── viz_simplex3d.py                               # 3D simplex viz panel (find_fixed_point.py only; needs matplotlib)
├── viz_radial.py                                  # Radial/spoke viz panel (find_fixed_point.py only)
├── viz_scatter_matrix.py                          # Scatter-plot-matrix viz panel (find_fixed_point.py only)
├── viz_parallel_coords.py                         # Parallel-coordinates viz panel (find_fixed_point.py only)
├── test_app.py                                    # Pytest + Playwright end-to-end tests
└── requirements.txt                               # Python dependencies
```
