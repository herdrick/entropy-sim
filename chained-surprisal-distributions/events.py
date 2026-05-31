import scipy.stats
import numpy as np

FAMILIES = {
    "Uniform": {
        "params": [
            {"name": "low",  "start": -10, "end": 10, "value":  0.0, "step": 0.1},
            {"name": "high", "start": -10, "end": 10, "value":  1.0, "step": 0.1},
        ],
        "fn": lambda p, n: scipy.stats.uniform.rvs(
            loc=p["low"], scale=max(p["high"] - p["low"], 1e-6), size=n
        ),
    },
    "Normal": {
        "params": [
            {"name": "μ", "start": -10, "end": 10,  "value": 0.0, "step": 0.1},
            {"name": "σ", "start":  0.1, "end": 10, "value": 1.0, "step": 0.1},
        ],
        "fn": lambda p, n: scipy.stats.norm.rvs(loc=p["μ"], scale=p["σ"], size=n),
    },
    "Beta": {
        "params": [
            {"name": "α", "start": 0.1, "end": 10, "value": 2.0, "step": 0.1},
            {"name": "β", "start": 0.1, "end": 10, "value": 5.0, "step": 0.1},
        ],
        "fn": lambda p, n: scipy.stats.beta.rvs(p["α"], p["β"], size=n),
    },
    "Exponential": {
        "params": [
            {"name": "λ", "start": 0.1, "end": 10, "value": 1.0, "step": 0.1},
        ],
        "fn": lambda p, n: scipy.stats.expon.rvs(scale=1.0 / p["λ"], size=n),
    },
}

FAMILY_NAMES = list(FAMILIES.keys())


def get_events(n, family, params):
    return FAMILIES[family]["fn"](params, n)
