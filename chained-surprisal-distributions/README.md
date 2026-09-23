*this README is mostly hand-written by a human*

# Entropy & Surprisal Explorer

This was made to explore the idea that the surprisal of an event is itself an event. The probability distribution of those surprisal events can itself be used as another model under which you can calculate surprisals of events. So, you can chain together such distributions and pass batches of events into the top of that chain, and get distributions of surprisals based on distributions of surprisals, and so on. You can play with that directly with iterated_surprisal_distributions.py.  

Invariably a fixed point is found. (Why is obvious enough when you think about it.) You can explore that with find_fixed_point.py.

There are continuous-space version of those two, but the results you get are just quirks in whatever density-fitting technique you use.

## Installation

```bash
pip install -r requirements.txt
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

`test_app.py` uses `pytest` + Playwright. 

```bash
pip install pytest-playwright
playwright install chromium
pytest test_app.py            # headless
pytest test_app.py --headed   # watch in a real browser
```
