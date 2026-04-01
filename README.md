# Entropy & surprisal simulations

Interactive tools for building intuition about Shannon entropy and surprisal. 

This repository contains three independent applications. See the README of each.

---

## Apps

### [new-distribution-with-each-event/](./new-distribution-with-each-event/README.md) — Interactive entropy simulator

A desktop GUI (matplotlib/TkAgg).

**Stack:** Python, numpy, scipy, matplotlib

#### [new-distribution-with-each-event/web/](./new-distribution-with-each-event/web/README.md) 
Web version.

---

### [chained-surprisal-distributions/](./chained-surprisal-distributions/README.md) — Entropy & surprisal explorer

A browser-based [Bokeh](https://bokeh.org) server app for building probability distributions from scratch. Generate events, bin them however you like by adding bin edges, and watch Shannon entropy update in the chart title with each change. Outermost bins extend to infinity; empty bins receive Laplace smoothing so the distribution is always well-defined.

**Stack:** Python, numpy, scipy, [Bokeh](https://bokeh.org) (browser UI); Playwright (optional tests)