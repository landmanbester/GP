"""
Computes the marginal posterior (or evidence) of GPR needed to train the the hyperparameters
"""

import numpy as np

class evidence(object):
    def __init__(self, x, y, kernel, mode="Full", prior_mean=None, XX=None, Phi=None, PhiTPhi=None, s=None, grid_regular=False):
        """
        :param x: inputs
        :param y: Nx1 vector of data 
        :param kernel: a class holding the covariance function and spectral densities
        :param mode:  whether to use full or RR GPR
        :param XX: absolute differences of inputs
        :param Phi: basis functions evaluated at x
        :param PhiTPhi: the product dot(Phi.T, Phi)
        :param s: square root of eigenvalues
        :param grid_regular: whether x is on a regular grid or not
        """
        self.x = x
        self.kernel = kernel
        self.mode = mode
        self.N = self.x.shape[0]
        if prior_mean is None:
            yp = np.zeros(self.N)
        else:
            yp = prior_mean(x)
        self.yDat = y - yp
        self.yTy = np.dot(self.yDat.T, self.yDat)
        if self.mode == "Full":
            self.XX = XX
        elif self.mode == "RR":
            self.s = s
            self.m = PhiTPhi.shape[0]
            self.Phi = Phi
            self.grid_regular = grid_regular
            self.PhiTPhi = PhiTPhi
        else:
            raise Exception('Mode %s not supported yet' % self.mode)


    def logL(self, theta):
        """
        Computes the negative log marginal posterior (hopefully in a way that saves memory?)
        :param theta: hypers 
        :return logp, dlogp: the negative log marginal posterior and its derivative w.r.t. hypers
        """
        if self.mode == "Full":
            # tmp is Ky
            tmp = self.kernel.cov_func(theta, self.XX, noise=True)
            # tmp is L
            try:
                tmp = np.linalg.cholesky(tmp)
            except:
                logp = 1.0e8
                dlogp = np.ones(theta.size) * 1.0e8
                return logp, dlogp
            detK = 2.0 * np.sum(np.log(np.diag(tmp)))
            # tmp is Linv
            tmp = np.linalg.inv(tmp)
            # tmp2 is Linvy
            tmp2 = np.dot(tmp, self.yDat)
            logp = np.dot(tmp2.T, tmp2) / 2.0 + detK / 2.0 + self.N * np.log(2 * np.pi) / 2.0
            nhypers = theta.size
            dlogp = np.zeros(nhypers)
            # tmp is Kinv
            tmp = np.dot(tmp.T, tmp)
            # tmp2 becomes Kinvy
            tmp2 = np.reshape(np.dot(tmp, self.yDat), (self.N, 1))
            # tmp2 becomes aaT
            tmp2 = np.dot(tmp2, tmp2.T)
            # tmp2 becomes Kinv - aaT
            tmp2 = tmp - tmp2
            dKdtheta = self.kernel.dcov_func(theta, self.XX, mode=0)
            dlogp[0] = np.sum(np.einsum('ij,ji->i', tmp2, dKdtheta)) / 2.0 #computes only the diagonal matrix product
            dKdtheta = self.kernel.dcov_func(theta, self.XX, mode=1)
            dlogp[1] = np.sum(np.einsum('ij,ji->i', tmp2, dKdtheta)) / 2.0
            dKdtheta = self.kernel.dcov_func(theta, self.XX, mode=2)
            dlogp[2] = np.sum(np.einsum('ij,ji->i', tmp2, dKdtheta)) / 2.0
            return logp, dlogp
        elif self.mode == "RR":
            S = self.kernel.spectral_density(theta, self.s)
            Lambdainv = np.diag(1.0 / S)
            Z = self.PhiTPhi + theta[2] ** 2 * Lambdainv
            try:
                L = np.linalg.cholesky(Z)
            except:
                print "Had to add jitter, theta = ", theta
                L = np.linalg.cholesky(Z + 1.0e-6)
                #
                # logp = 1.0e2
                # dlogp = np.ones(theta.size)*1.0e8
                # return logp, dlogp
            Linv = np.linalg.inv(L)
            Zinv = np.dot(Linv.T, Linv)
            logdetZ = 2.0 * np.sum(np.log(np.diag(L)))
            # Get the log term
            logQ = (self.N - self.m) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
            # Get the quadratic term
            PhiTy = np.dot(self.Phi.T, self.yDat)
            ZinvPhiTy = np.dot(Zinv, PhiTy)
            yTQinvy = (self.yTy - np.dot(PhiTy.T, ZinvPhiTy)) / theta[2] ** 2
            # Get their derivatives
            dlogQdtheta = np.zeros(theta.size)
            dyTQinvydtheta = np.zeros(theta.size)
            for i in xrange(theta.size - 1):
                dSdtheta = self.kernel.dspectral_density(theta, S, self.s, mode=i)
                dlogQdtheta[i] = np.sum(dSdtheta / S) - theta[2] ** 2 * np.sum(dSdtheta / S * np.diag(Zinv) / S)
                dyTQinvydtheta[i] = -np.dot(ZinvPhiTy.T, np.dot(np.diag(dSdtheta / S ** 2), ZinvPhiTy))
            # Get derivatives w.r.t. sigma_n
            dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.m) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
            dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, np.dot(Lambdainv, ZinvPhiTy)) - yTQinvy) / theta[2]

            logp = (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
            dlogp = (dlogQdtheta + dyTQinvydtheta) / 2
            return logp, dlogp
        else:
            raise Exception('Mode %s not supported yet' % self.mode)