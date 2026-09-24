# Entropy & surprisal simulations

Interactive tools for building intuition about Shannon entropy and surprisal. 

This repository contains three independent applications. See the README of each.

---

## Apps

### [new-distribution-with-each-event/](./new-distribution-with-each-event/README.md) — Interactive entropy simulator

Made to explore changes in surprisals and entropy as the distribution (under which the surprisals are calculated) changes.

**Stack:** Three.js, Chart.js, jStat (browser, no build step)

An earlier Python/matplotlib version is archived at [archive/new-distribution-with-each-event/](./archive/new-distribution-with-each-event/README.md).

---

### [chained-surprisal-distributions/](./chained-surprisal-distributions/README.md) — Entropy & surprisal explorer

Made to explore the idea that the surprisal of an event is itself an event.

**Stack:** Python, numpy, scipy, [Bokeh](https://bokeh.org) (browser UI); Playwright (optional tests)
