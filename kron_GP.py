#!/usr/bin/env python
"""
This class performs GPR on a regular cartesian grid in O(N) time where N = N_1 x N_2 X ... x N_D

So far only the vanilla implementation is available, still need to figure out how to implement RR and SS versions in 
a consistent way
"""

import numpy as np
from scipy import optimize as opt
from GP.tools import posterior_mean, posterior_covariance, marginal_posterior, kronecker_tools, abs_diff

class KronGP(object):
    def __init__(self, x, xp, y, Sigmay=None, covariance_function='sqexp'):
        """
        :param x: an array of arrays holding inputs i.e. x = [[x_1], [x_2], ..., [x_3]] where x_1 in R^{N_1} etc.
        :param xp: an array of arrays holding targets i.e. xp = [[xp_1], [xp_2], ..., [xp_3]] where xp_1 in R^{Np_1} etc.
        :param y: an array of arrays holding data i.e. y = [[y_1], [y_2], ..., [y_3]] where y_1 in R^{N_1} etc.
        :param Sigmay: an array of arrays holding variances i.e. Sigmay = [[Sigmay_1], [Sigmay_2], ..., [Sigmay_3]] 
                where Sigmay_1 in R^{N_1} etc.
        :param covariance_function: a list holding the covariance functions to use for each dimension
        """
        self.D = x.shape[0]
        self.x = x
        self.xp = xp
        self.y = y
        if Sigmay is not None:
            self.Sigmay = Sigmay
        else:
            self.Sigmay = None
        self.covariance_function = covariance_function

        # initialise the individual diff square grids
        self.XX = []
        self.XXp = []
        self.XXpp = []
        for i in xrange(self.D):
            self.XX.append(abs_diff.abs_diff(self.x[i], self.x[i]))
            self.XXp.append(abs_diff.abs_diff(self.x[i], self.xp[i]))
            self.XXpp.append(abs_diff.abs_diff(self.xp[i], self.xp[i]))
        self.XX = np.asarray(self.XX)
        self.XXp = np.asarray(self.XXp)
        self.XXpp = np.asarray(self.XXpp)

        # set the kernels
        from GP.kernels import exponential_squared  # only using this one for now
        self.kernels = []
        for i in xrange(self.D):
            if self.Sigmay is not None:
                self.kernels.append(exponential_squared.sqexp(Sigmay=self.Sigmay[i]))
            else:
                self.kernels.append(exponential_squared.sqexp())