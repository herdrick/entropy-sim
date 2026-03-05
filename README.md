# Entropy & surprisal simulations

Interactive tools for building intuition about Shannon entropy and surprisal. These apps let you watch information-theoretic quantities update live as data arrives, making the math tangible rather than abstract.

This repository contains two independent applications. Each has its own full README with installation and usage instructions.

---

## Apps

### [start-with-simple-prompt/](./start-with-simple-prompt/README.md) — Interactive entropy simulator

A desktop GUI (matplotlib/TkAgg) that streams samples from one of six hidden probability distributions. Four live panels update as samples arrive: an empirical histogram, an entropy-over-time trace, a per-event surprisal stream, and a latest-event marker. You try to guess the mystery source, then reveal the true distribution and theoretical entropy to check your intuition.

**Stack:** Python, numpy, scipy, matplotlib

---

### [start-with-detailed-prompt/](./start-with-detailed-prompt/README.md) — Entropy & surprisal explorer

A browser-based Bokeh server app for building probability distributions from scratch. Generate events, bin them however you like by adding fenceposts, and watch Shannon entropy update in the chart title with each change. Outermost bins extend to infinity; empty bins receive Laplace smoothing so the distribution is always well-defined.

**Stack:** Python, numpy, scipy, Bokeh (browser UI); Playwright (optional tests)

---

## Getting started

Each app is self-contained. Navigate into the subdirectory that interests you and follow its README:

```bash
# Desktop simulator
cd start-with-simple-prompt
# then follow start-with-simple-prompt/README.md

# Browser-based explorer
cd start-with-detailed-prompt
# then follow start-with-detailed-prompt/README.md
```

Both apps require Python 3.8 or later and no API keys or external accounts.
