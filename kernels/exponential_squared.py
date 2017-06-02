"""
The exponential squared covariance function, its derivative and spectral density
"""

import numpy as np

def cov_func(theta, x, noise=True):
    """
    Covariance function possibly including noise variance
    :param theta: hypers
    :param x: the inputs x[i] - x[j] (usually matrix of differences)
    :param noise: whether to add noise or not
    """
    if not noise:
        # Squared exponential
        return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))
    else:
        # Squared exponential
        return theta[0] ** 2 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * np.eye(x.shape[0])


def dcov_func(theta, x, mode=0):
    """
    Derivates of the covariance function w.r.t. the hyperparameters
    :param theta: hypers
    :param x: the inputs x[i] - x[j] (usually matrix of differences)
    :param mode: use to determine which hyper we are taking the derivative w.r.t.
    """
    if mode == 0:
        return 2 * theta[0] * np.exp(-x ** 2 / (2 * theta[1] ** 2))
    elif mode == 1:
        return x ** 2 * theta[0] ** 2 * np.exp(-x ** 2 / (2 * theta[1] ** 2)) / theta[1] ** 3
    elif mode == 2:
        return 2 * theta[2] * np.eye(x.shape[0])

def spectral_density(theta, s):
    """
    The spectral density of the squared exponential covariance function
    :param theta: hypers
    :param s: Fourier dual of x
    """
    return theta[0]**2.0*np.sqrt(2.0*np.pi*theta[1]**2)*np.exp(-theta[1]**2*s**2/2)

def dspectral_density(theta, S, s, mode=0):
    """
    Derivative of the spectral density corresponding to the squared exponential covariance function
    :param theta: hypers
    :param S: value of spectral density
    :param s: Fourier dual of x
    :param mode: use to determine which hyper we are taking the derivative w.r.t.
    """
    if mode == 0:
        return 2 * S / theta[0]
    elif mode == 1:
        return S/theta[1] - s**2*theta[1]*S