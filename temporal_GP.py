#!/usr/bin/env python
"""
This class performs temporal GPR (i.e. GPR in 1D)
"""

import numpy as np
from scipy import optimize as opt
from GP.tools import posterior_mean, posterior_covariance, marginal_posterior

class TemporalGP(object):
    def __init__(self, x, xp, y, covariance_function='sqexp', mode="Full", nu=2.5, basis="Rect", M=12, L=5.0, grid_regular=False):
        """
        :param x: vector of inputs
        :param xp: vector of targets
        :param y: vector of data
        :param covariance_function: the kind of covariance function to use (currently 'sqexp' or 'mattern')
        :param mode: whether to do a full GPR or a reduced rank GPR (currently "Full" or "RR")
        :param nu: specifies the kind of Mattern function to use (must be a half integer)
        :param basis: specifies the class of basis functions to use (currently only "Rect")
        :param M: integer specifying the number of basis functions to use
        :param L: float specifying the boundary of the domain 
        """
        # set inputs and targets
        self.N= x.size
        self.Np = xp.size
        self.D = 1
        self.x = x
        self.xp = xp
        self.y = y

        # set covariance
        if covariance_function=="sqexp":
            from GP.kernels import exponential_squared
            # Initialise kernel
            self.kernel = exponential_squared.sqexp()
        elif covariance_function=="mattern":
            from GP.kernels import mattern
            # Initialise kernel
            self.kernel = mattern.mattern(p=int(nu))
        else:
            raise Exception('Kernel %s not supported yet' % covariance_function)

        # Get inputs required to evaluate covariance function
        self.mode = mode
        if mode=="Full": # here we need the differences squared
            from GP.tools import abs_diff
            # Initialise absolute differences
            self.XX = abs_diff.abs_diff(x, x)
            self.XXp = abs_diff.abs_diff(x, xp)
            self.XXpp = abs_diff.abs_diff(xp, xp)
            # Instantiate posterior mean, cov and evidence classes
            self.meano = posterior_mean.meanf(self.x, self.xp, self.y, self.kernel, mode=self.mode, XX=self.XX, XXp=self.XXp)
            self.meanf = lambda theta : self.meano.give_mean(theta)
            self.covo = posterior_covariance.covf(self.kernel, mode=self.mode, XX=self.XX, XXp=self.XXp, XXpp=self.XXpp)
            self.covf = lambda theta: self.covo.give_covaraince(theta)
            self.logpo = marginal_posterior.evidence(self.x, self.yDat, self.kernel, mode=self.mode, XX=self.XX)
            self.logp = lambda theta: self.logpo.logL(theta)
        elif mode=="RR": # here we need to evaluate the basis functions
            from GP.tools import make_basis
            # check consistency of inputs
            if np.asarray(M).size == 1: # 1D so only a single value allowed
                self.M = np.asarray(M)
            else:
                raise Exception('Inconsistent dimensions specified for M. Must be 1D.')
            if np.asarray(L).size == 1: # 1D so only a single value allowed
                self.L = np.asarray(L)
            else:
                raise Exception('Inconsistent dimensions specified for L. Must be 1D.')
            # Construct the basis functions
            if basis=="Rect":
                from GP.basisfuncs import rectangular
                self.Phi = make_basis.get_eigenvectors(self.x, self.M, self.L, rectangular.phi)
                self.Phip = make_basis.get_eigenvectors(self.xp, self.M, self.L, rectangular.phi)
                self.Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)
            else:
                print "%s basis functions not supported yet"%basis

            # Precompute some terms that never change
            # If inputs are on a regular grid we only need the diagonal
            self.grid_regular = grid_regular
            if self.grid_regular:
                self.PhiTPhi = np.einsum('ij,ji->i', self.Phi.T, self.Phi)
            else:
                self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
            self.s = np.sqrt(self.Lambda)

            self.meano = posterior_mean.meanf(self.x, self.xp, self.y, self.kernel, mode=self.mode, Phi=self.Phi,
                                        Phip=self.Phip, PhiTPhi=self.PhiTPhi, s=self.s, grid_regular=self.grid_regular)
            self.meanf = lambda theta : self.meano.give_mean(theta)
            self.covo = posterior_covariance.covf(self.kernel, mode=self.mode, Phi=self.Phi, Phip=self.Phip,
                                                  PhiTPhi=self.PhiTPhi, s=self.s, grid_regular=self.grid_regular)
            self.covf = lambda theta: self.covo.give_covaraince(theta)
            self.logpo = marginal_posterior.evidence(self.x, self.y, self.kernel, mode=self.mode, Phi=self.Phi,
                                                     PhiTPhi=self.PhiTPhi, s=self.s, grid_regular=self.grid_regular)
            self.logp = lambda theta: self.logpo.logL(theta)
        else:
            raise Exception('Mode %s not supported yet'%mode)


    def train(self, theta0, bounds=None):

        if bounds is None:
            # Set default bounds for hypers (they must be strictly positive)
            bnds = ((1e-7, None), (1e-7, None), (1e-7, None))
        else:
            bnds = bounds

        # Do optimisation
        thetap = opt.fmin_l_bfgs_b(self.logp, theta0, fprime=None, bounds=bnds) #, factr=1e10, pgtol=0.1)

        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return thetap[0]