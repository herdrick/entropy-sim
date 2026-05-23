import scipy.stats
import numpy as np

SOURCES = {
    "Uniform(0, 1)":          lambda n: scipy.stats.uniform.rvs(size=n),
    "Normal(0, 1)":           lambda n: scipy.stats.norm.rvs(loc=0, scale=1, size=n),
    "Normal(0, 3)":           lambda n: scipy.stats.norm.rvs(loc=0, scale=3, size=n),
    "Bimodal Normal":         lambda n: np.where(
                                  np.random.random(n) < 0.5,
                                  scipy.stats.norm.rvs(loc=-3, scale=1, size=n),
                                  scipy.stats.norm.rvs(loc=3,  scale=1, size=n),
                              ),
    "Beta(2, 5)":             lambda n: scipy.stats.beta.rvs(2, 5, size=n),
    "Beta(0.5, 0.5)":         lambda n: scipy.stats.beta.rvs(0.5, 0.5, size=n),
}

SOURCE_NAMES = list(SOURCES.keys())


def get_events(n, source=SOURCE_NAMES[0]):
    return SOURCES[source](n)
