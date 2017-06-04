#!/usr/bin/env python
"""
This class performs spatial GPR (i.e. GPR in ND)
"""

import numpy as np
from scipy import optimize as opt


class SpatialGP(object):
    def __init__(self, x, xp, covariance_function='sqexp', mode="Full", nu=2.5, basis="Rect", M=12, L=5.0):
        """
        :param x: vector of inputs
        :param xp: vector of targets
        :param covariance_function: the kind of covariance function to use (currently 'sqexp' or 'mattern')
        :param mode: whether to do a full GPR or a reduced rank GPR (currently "Full" or "RR")
        :param nu: specifies the kind of Mattern function to use (must be a half integer)
        :param basis: specifies the class of basis functions to use (currently only "Rect")
        """
        # set inputs and targets
        try:
            self.N, self.D = x.shape
            self.Np = xp.shape[0]
        except:
            self.N = x.size
            self.D = 1
            self.Np = xp.size
        self.x = x
        self.xp = xp

        # set covariance
        if covariance_function == "sqexp":
            from GP.kernels import exponential_squared
            # Initialise kernel
            self.kernel = exponential_squared.sqexp()
        elif covariance_function == "mattern":
            from GP.kernels import mattern
            # Initialise kernel
            self.kernel = mattern.mattern(p=int(nu))

        # Get inputs required to evaluate covariance function
        self.mode = mode
        if mode == "Full":  # here we need the differences squared
            from GP.tools import abs_diff
            # Initialise absolute differences
            self.XX = abs_diff.abs_diff(x, x)
            self.XXp = abs_diff.abs_diff(x, xp)
            self.XXpp = abs_diff.abs_diff(xp, xp)
        elif mode == "RR":  # here we need to evaluate the basis functions
            from GP.tools import make_basis
            # check consistency of inputs
            if np.asarray(
                    M).size == 1:  # if single value is specified for M we use the same number of basis functions for each dimension
                self.M = np.tile(M, self.D)
            elif np.asarray(M).size == self.D:
                self.M = M
            else:
                raise Exception('Inconsistent dimensions specified for M')
            if np.asarray(
                    L).size == 1:  # if single value is specified for L we use square domain (hopefully also circular in future)
                self.L = np.tile(L, self.D)
            elif np.asarray(L).size == self.D:
                self.L = L
            else:
                raise Exception('Inconsistent dimensions specified for L')
            # Construct the basis functions
            if basis == "Rect":
                from GP.basisfuncs import rectangular
                self.Phi = make_basis.get_eigenvectors(self.x, self.M, self.L, rectangular.phi)
                self.Phip = make_basis.get_eigenvectors(self.xp, self.M, self.L, rectangular.phi)
                self.Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)
            else:
                print "%s basis functions not supported yet" % basis

            # Precompute some terms that never change
            self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
            self.s = np.sqrt(self.Lambda)


