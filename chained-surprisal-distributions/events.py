import scipy.stats
import numpy as np


def get_events(n):
    return scipy.stats.uniform.rvs(size=n)
