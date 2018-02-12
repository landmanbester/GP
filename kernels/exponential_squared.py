"""
The exponential squared covariance function, its derivative and spectral density
"""

import numpy as np
from GP.tools import kronecker_tools as kt

class sqexp(object):
    def __init__(self, p=None, D=1, Sigmay=None, mode=None):
        # p is not relevant to sq exp
        # Check that D is an integer
        if not isinstance(D, (int, long)):
            print "Converting D to integer"
            D = int(D)
        self.D = D
        if Sigmay is not None:
            self.Sigmay = Sigmay
        else:
            self.Sigmay = None
        self.mode = mode


    def cov_func(self, theta, x, noise=True, i_dim=None):
        """
        Covariance function possibly including noise variance
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param noise: whether to add noise or not
        """
        if noise:
            if self.Sigmay is not None:
                if i_dim is None:
                    return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * self.Sigmay
                else:
                    return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * self.Sigmay[i_dim]
            else:
                return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * np.eye(x.shape[0])
        else:
            return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))

    def dcov_func(self, theta, x, K, mode=None):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param mode: use to determine which hyper we are taking the derivative w.r.t.
        """
        if mode is not None:
            if mode == "sigmaf":
                return 2 * K / theta[0]
            elif mode == "l":
                return x ** 2 * K / theta[1] ** 3
            elif mode == "sigman":
                if self.Sigmay is not None:
                    if self.mode == "kron":
                        return 2 * theta[2] * self.Sigmay
                    else:
                        return 2 * theta[2] * np.diag(self.Sigmay)
                else:
                    if self.mode == "kron":
                        return 2 * theta[2] * kt.kron_collapse_shape(x)  # broadcasting to shape of Sigmay, boradcast to shape eye(N) using Kronecker tools
                    else:
                        return 2 * theta[2] * np.eye(x.shape[0])
        else:
            raise Exception('You have to specify a mode to evaluate dcov_func')

    def spectral_density(self, theta, s):
        """
        The spectral density of the squared exponential covariance function
        :param theta: hypers
        :param s: Fourier dual of x
        """
        return np.sqrt(2*np.pi)*theta[0]**2.0*(theta[1]**2)**(self.D/2.0)*np.exp(-theta[1]**2*s**2/2.0)

    def dspectral_density(self, theta, S, s, mode=0):
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
            return self.D*S/theta[1] - s**2*theta[1]*S