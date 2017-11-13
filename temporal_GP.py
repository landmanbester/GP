#!/usr/bin/env python
"""
This class performs temporal GPR (i.e. GPR in 1D)
"""

import numpy as np
from scipy import optimize as opt
from GP.tools import posterior_mean, posterior_covariance, marginal_posterior

class TemporalGP(object):
    def __init__(self, x, xp, y, Sigmay=None, prior_mean=None, covariance_function='sqexp', mode="Full", nu=2.5, basis="Rect", M=12, L=5.0, grid_regular=False):
        """
        :param x: vector of inputs
        :param xp: vector of targets
        :param y: vector of data
        :param prior_mean: a prior mean function (must be callable)
        :param covariance_function: the kind of covariance function to use (currently 'sqexp' or 'mattern')
        :param mode: whether to do a full GPR or a reduced rank GPR (currently "Full" or "RR")
        :param nu: specifies the kind of Mattern function to use (must be a half integer)
        :param basis: specifies the class of basis functions to use (currently only "Rect")
        :param M: integer specifying the number of basis functions to use
        :param L: float specifying the boundary of the domain 
        """
        # set inputs and targets enforcing correct sizes
        self.N = x.shape[0]
        try:
            self.D = x.shape[1]
            self.x = x
        except:
            self.D = 1
            self.x = x.reshape(self.N, self.D)
        if self.D > 1:
            raise Exception("Must pass 1D input to temporal_GP")
        self.Np = xp.size
        self.xp = xp.reshape(self.Np, self.D)
        self.y = y.reshape(self.N, self.D)

        # check that prior mean is callable and set it
        try:
            ym = prior_mean(self.x)
            self.prior_mean = prior_mean
            del ym
        except:
            raise Exception("Prior mean function must be callable")

        # set covariance
        if covariance_function=="sqexp":
            from GP.kernels import exponential_squared
            # Initialise kernel
            self.kernel = exponential_squared.sqexp(Sigmay)
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
            self.meano = posterior_mean.meanf(self.x, self.xp, self.y, self.kernel, mode=self.mode,
                                              prior_mean=self.prior_mean, XX=self.XX, XXp=self.XXp)
            self.meanf = lambda theta: self.meano.give_mean(theta)
            self.covo = posterior_covariance.covf(self.kernel, mode=self.mode, XX=self.XX, XXp=self.XXp, XXpp=self.XXpp)
            self.covf = lambda theta: self.covo.give_covariance(theta)
            self.logpo = marginal_posterior.evidence(self.x, self.y, self.kernel, mode=self.mode, XX=self.XX)
            self.logp = lambda theta: self.logpo.logL(theta)
        elif mode=="RR": # here we need to evaluate the basis functions
            from GP.tools import make_basis
            # check consistency of inputs
            if np.asarray(M).size == 1: # 1D so only a single value allowed
                self.M = np.array([M])  # cast as array to make it compatible with make_basis class
            else:
                raise Exception('Inconsistent dimensions specified for M. Must be 1D.')
            if np.asarray(L).size == 1: # 1D so only a single value allowed
                self.L = np.array([L]) # cast as array to make it compatible with make_basis class
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
                self.PhiTPhi = np.einsum('ij,ji->i', self.Phi.T, self.Phi) # computes only the diagonal entries
            else:
                self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
            self.s = np.sqrt(self.Lambda)

            self.meano = posterior_mean.meanf(self.x, self.xp, self.y, self.kernel, mode=self.mode, Phi=self.Phi,
                                        Phip=self.Phip, PhiTPhi=self.PhiTPhi, s=self.s, grid_regular=self.grid_regular)
            self.meanf = lambda theta : self.meano.give_mean(theta)
            self.covo = posterior_covariance.covf(self.kernel, mode=self.mode, Phi=self.Phi, Phip=self.Phip,
                                                  PhiTPhi=self.PhiTPhi, s=self.s, grid_regular=self.grid_regular)
            self.covf = lambda theta: self.covo.give_covariance(theta)
            self.logpo = marginal_posterior.evidence(self.x, self.y, self.kernel, mode=self.mode, Phi=self.Phi,
                                                        PhiTPhi=self.PhiTPhi, s=self.s, grid_regular=self.grid_regular)
            self.logp = lambda theta: self.logpo.logL(theta)
        else:
            raise Exception('Mode %s not supported yet'%mode)

        self.has_posterior = False #flag to indicate if the posterior has been computed

    def train(self, theta0, bounds=None):
        """
        Trains the hypers using built in scipy optimizer
        :param theta0: initial guess for hypers
        :param bounds: optional set min and max bounds on hypers
        :return: the optimised hyperparameters
        """
        if bounds is None:
            if self.mode=="Full":
                # Set default bounds for hypers (they must be strictly positive)
                bnds = ((1e-6, None), (1e-6, None), (1e-6, None))
            elif self.mode=="RR":
                # this keeps l within a reasonable range
                bnds = ((1e-6, None), ((self.L[0] + 0.0)/self.M[0], 0.9*self.L[0]), (1e-6, None))
        else:
            bnds = bounds # make sure above criteria on bounds satisfied if passing default bounds

        # Do optimisation
        thetap = opt.fmin_l_bfgs_b(self.logp, theta0, fprime=None, bounds=bnds) #, factr=1e10, pgtol=0.1)

        theta = thetap[0]

        if self.mode=="RR":
            if theta[1] < 2*self.L/self.M:
                print "Warning l < 2L/M so result might be inaccurate. Consider increasing M (or decreasing L if possible)"
            elif theta[1] > 0.6*self.L:
                print "Warning l > 0.6*L so result might be inaccurate. Consider increasing L"


        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return theta

    def set_posterior(self, theta):
        """
        Set the posterior mean and covariance functions and compute the eigen-decomposition required to draw samples
        from it.
        :param theta: optimised value of hypers 
        """
        if not self.has_posterior:
            self.post_mean = self.meanf(theta).reshape(self.Np, 1)  # reshaping for compatibility with + in draw_samps
            if self.mode == "Full":
                # Get the posterior mean function
                self.post_cov = self.covf(theta)
                self.W, self.V = np.linalg.eigh(self.post_cov)
                I = np.argwhere(self.W < 0.0)
                self.W[I] = 0.0
                self.sqrtW = np.nan_to_num(np.sqrt(np.nan_to_num(self.W))).reshape(self.Np, 1) # reshaping for compatibility with * in draw_samps
            elif self.mode == "RR":
                self.post_mean_coeffs = self.meano.give_RR_coeffs(theta)
                self.post_cov_coeffs = self.covo.give_RR_covcoeffs(theta) #not computing this for memory sake

            self.has_posterior = True
        else:
            print "Posterior has been set already"


    def draw_samps(self, Nsamps, theta, meanf=None, covf=None):
        """
        Draws N samples from the posterior over the GP. Only use after training
        :param Nsamps: number of samples
        :param theta: hypers with which to draw samples
        :param meanf: posterior mean function (pass prior mean function to draw samps from the prior)
        :param covf: posterior covariance matrix (pass prior covariance function to draw samps from the prior)
        """
        # set posterior if nothing is passed in
        if covf is None and meanf is None:
            self.set_posterior(theta)

        if self.mode=="Full":
            # Draw random seeds
            u = np.random.randn(self.Np, Nsamps)
            return self.post_mean + self.V.dot(self.sqrtW*u)
        elif self.mode=="RR":
            # In this case samples of the parameters are completely equivalent to samples of function
            samps_coeffs = np.random.multivariate_normal(self.post_mean_coeffs.ravel(), self.post_cov_coeffs, Nsamps).squeeze().T #might want to do this more efficiently
            return np.dot(self.Phip, samps_coeffs)