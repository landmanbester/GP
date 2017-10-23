#!/usr/bin/env python
"""
A function to draw samples from GP with specified kernel
"""

import numpy as np
from GP.tools import abs_diff

def draw_samples(meanf, x, theta, cov_func, N):
    XX = abs_diff.abs_diff(x, x)
    covf = cov_func(XX, theta)
    samps = np.random.multivariate_normal(meanf, covf, N)
    return samps
