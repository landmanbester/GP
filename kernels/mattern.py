"""
The Mattern covariance function for p an integer, its derivative and spectral density
"""

import numpy as np
from scipy.special import gamma
from scipy.misc import factorial

class mattern(object):
    def __init__(self, p=3, D=1):
        # Check that p is an integer
        if not isinstance(p, (int, long)):
            print "Converting p to integer"
            p = int(p)
        self.p = p
        # Check that D is an integer
        if not isinstance(D, (int, long)):
            print "Converting D to integer"
            D = int(D)
        self.D = D
        # Precompute terms that stay the same
        self.v = p + 0.5
        self.I = np.arange(p+1) #for vectorising
        self.factorial_arr = factorial(p + self.I)/(factorial(self.I)*factorial(p - self.I))
        self.gamma_term = gamma(p+1)/gamma(2*p+1)
        self.root2v = np.sqrt(2*self.v)

        # And for the spectral density
        self.preSterm = 2**D*np.pi**(D/2.0)*gamma(self.v + D/2.0)*(2*self.v)**self.v/gamma(self.v)



    def cov_func(self, theta, x, noise=True):
        """
        Covariance function possibly including noise variance
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param noise: whether to add noise or not
        """
        tmp = np.tile(2 * self.root2v * x.flatten() / theta[1], (self.p + 1, 1)).T ** (self.p - self.I)
        K = (theta[0]**2.0*np.exp(-self.root2v*x.flatten()/theta[1])*self.gamma_term*np.sum(self.factorial_arr*tmp,
                                                                               axis=1)).reshape(x.shape)
        if noise:
            return K + theta[2] ** 2.0 * np.eye(x.shape[0])
        else:
            return K

    def cov_func2(self, theta, x, noise=True):
        if not noise:
            return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2))
        else:
            return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (
            1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2)) + theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func(self, theta, x, K, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param K: the value of the covariance function
        :param mode: use to determine which hyper we are taking the derivative w.r.t.
        """
        if mode == 0:
            return 2 * K / theta[0]
        elif mode == 1:
            tmp = np.tile(2 * self.root2v * x.flatten() / theta[1], (self.p + 1, 1)).T ** (self.p - self.I)
            return self.root2v*x*K/theta[1]**2 + (theta[0]**2.0*np.exp(-self.root2v*x.flatten()/theta[1])*self.gamma_term*\
                        np.sum(-self.factorial_arr*(self.p - self.I)*tmp/theta[1], axis=1)).reshape(x.shape)
        elif mode == 2:
            return 2 * theta[2] * np.eye(x.shape[0])

    def spectral_density(self, theta, s):
        """
        The spectral density of the squared exponential covariance function
        :param theta: hypers
        :param s: Fourier dual of x
        """
        return theta[0]**2.0*self.preSterm*(2*self.v/theta[1]**2+4*np.pi**2*s**2)**(-self.v-self.D/2.0)/theta[1]**(2*self.v)

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
            return -2*self.v*S/theta[1] + 4*self.v*(self.v + self.D/2.0)*S/(theta[1]**3*(2*self.v/theta[1]**2 + 4*np.pi**2*s**2))