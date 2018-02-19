"""
Computes the marginal posterior (or evidence) of GPR needed to train the the hyperparameters
"""

import numpy as np
from GP.tools import kronecker_tools as kt

class evidence(object):
    def __init__(self, x, y, kernel, Sigmay=None, mode="Full", prior_mean=None, XX=None, Phi=None, PhiTPhi=None, s=None, grid_regular=False):
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
        if self.mode != "kron":
            self.N = self.x.shape[0]
        else:
            self.N = kt.kron_N(self.x)
        self.Sigmay = Sigmay
        if prior_mean is None:
            yp = np.zeros([self.N, 1])
        else:
            yp = prior_mean(x).reshape(self.N, 1)
        self.yDat = y.reshape(self.N, 1) - yp
        if self.mode == "Full" or self.mode == "kron":
            self.XX = XX
        elif self.mode == "RR":
            self.yTy = np.dot(self.yDat.T, self.yDat)
            self.s = s
            self.M = PhiTPhi.shape[0]
            self.Phi = Phi
            self.grid_regular = grid_regular
            self.PhiTPhi = PhiTPhi
        else:
            raise Exception('Mode %s not supported yet' % self.mode)


    def logL(self, theta):
        """
        Computes the negative log marginal posterior
        :param theta: hypers 
        :return logp, dlogp: the negative log marginal posterior and its derivative w.r.t. hypers
        """
        if self.mode == "Full":
            # tmp is Ky
            Ky = self.kernel.cov_func(theta, self.XX, noise=True)
            # tmp is L
            try:
                L = np.linalg.cholesky(Ky)
            except:
                print "Had to add jitter, theta = ", theta
                F = True
                while F:
                    jit = 1e-6
                    try:
                        L = np.linalg.cholesky(Ky + jit*np.eye(self.N))
                        F = False
                    except:
                        jit *=10.0
                        F = True
            try:
                tmp = np.linalg.cholesky(Ky)
            except:
                logp = 1.0e8
                dlogp = np.ones(theta.size) * 1.0e8
                return logp, dlogp
            detK = 2.0 * np.sum(np.log(np.diag(tmp)))
            # tmp is Linv
            tmp = np.linalg.inv(tmp)
            # tmp2 is Linvy
            tmp2 = np.dot(tmp, self.yDat)
            logp = np.dot(tmp2.conj().T, tmp2).real / 2.0 + detK / 2.0 + self.N * np.log(2 * np.pi) / 2.0
            nhypers = theta.size
            dlogp = np.zeros(nhypers)
            # tmp is Kinv
            tmp = np.dot(tmp.T, tmp)
            # tmp2 becomes Kinvy
            tmp2 = np.reshape(np.dot(tmp, self.yDat), (self.N, 1))
            # tmp2 becomes aaT
            tmp2 = (np.dot(tmp2, tmp2.conj().T)).real
            # tmp2 becomes Kinv - aaT
            tmp2 = tmp - tmp2
            K = self.kernel.cov_func(theta, self.XX, noise=False)
            dKdtheta = self.kernel.dcov_func(theta, self.XX, K, mode=0)
            dlogp[0] = np.sum(np.einsum('ij,ji->i', tmp2, dKdtheta)) / 2.0 #computes only the diagonal matrix product
            dKdtheta = self.kernel.dcov_func(theta, self.XX, K, mode=1)
            dlogp[1] = np.sum(np.einsum('ij,ji->i', tmp2, dKdtheta)) / 2.0
            dKdtheta = self.kernel.dcov_func(theta, self.XX, K, mode=2)
            dlogp[2] = np.sum(np.einsum('ij,ji->i', tmp2, dKdtheta)) / 2.0
            return logp, dlogp
        elif self.mode == "RR":
            S = self.kernel.spectral_density(theta, self.s)
            if np.any(S < 1e-13):
                I = np.argwhere(S < 1e-13)
                S[I] += 1.0e-13
            Lambdainv = np.diag(1.0 / S)
            Z = self.PhiTPhi + theta[2] ** 2 * Lambdainv
            try:
                L = np.linalg.cholesky(Z)
            except:
                print "Had to add jitter, theta = ", theta
                F = True
                while F:
                    jit = 1e-6
                    try:
                        L = np.linalg.cholesky(Z + jit*np.eye(self.M))
                        F = False
                    except:
                        jit *=10.0
                        F = True
            Linv = np.linalg.inv(L)
            Zinv = np.dot(Linv.T, Linv)
            logdetZ = 2.0 * np.sum(np.log(np.diag(L)))
            # Get the log term
            logQ = (self.N - self.M) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
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
                dyTQinvydtheta[i] = -np.dot(ZinvPhiTy.T, dSdtheta / S * ZinvPhiTy.squeeze()/S)
            # Get derivatives w.r.t. sigma_n
            dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.M) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
            dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, ZinvPhiTy.squeeze()/S) - yTQinvy) / theta[2]

            logp = (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
            dlogp = (dlogQdtheta + dyTQinvydtheta) / 2
            return logp, dlogp
        elif self.mode == "kron":
            # get dims
            D = self.XX.shape[0]
            # broadcast theta (sigmaf and sigman is a shared hyperparameter but easiest to deal with this way)
            thetan = np.zeros([D, 3])
            thetan[:, 0] = theta[0]
            thetan[:, -1] = theta[-1]
            for i in xrange(D): # this is how many length scales will be involved in te problem
                thetan[i, 1] = theta[i+1] # assuming we set up the theta vector as [[sigmaf, l_1, sigman], [sigmaf, l_2, ..., sigman]]
            # get the Kronecker matrices
            K = []
            for i in xrange(D):
                K.append(self.kernel[i].cov_func(thetan[i], self.XX[i], noise=False))
            K = np.asarray(K)
            # do eigen-decomposition
            Lambdas, Qs = kt.kron_eig(K)
            QsT = kt.kron_transpose(Qs)
            # get alpha vector
            alpha = kt.kron_matvec(QsT[::-1], self.yDat)
            Lambda = kt.kron_diag(Lambdas)
            if self.Sigmay is not None:
                Lambda += theta[-1]**2*kt.kron_diag(self.Sigmay)  # absorb weights into Lambdas
            else:
                Lambda += theta[-1] ** 2 * np.ones(self.N)
            alpha = alpha / Lambda  # same as matrix product with inverse of diagonal
            alpha = kt.kron_matvec(Qs[::-1], alpha)
            # get negative log marginal likelihood
            logp = 0.5*(self.yDat.T.dot(alpha) + np.sum(np.log(Lambda)) + self.N*np.log(2.0*np.pi))
            # get derivatives
            dKdtheta = []
            # first w.r.t. sigmaf which only needs to be done once and will have same shape as K
            dKdtheta.append(self.kernel[0].dcov_func(thetan[0], self.XX, K, mode='sigmaf')) # doesn't matter which theta/kernel we pass because we have already evaluated K
            # now get w.r.t. length scales
            for i in xrange(D): # one length scale for each dimension
                dKdtheta.append(self.kernel[i].dcov_func(thetan[i], self.XX[i], K[i], mode='l')) # here it does matter, will be of shape K[i]
            # finally get deriv w.r.t sigman (also only need to do this once)
            dKdtheta.append(self.kernel[0].dcov_func(thetan[0], self.XX, K, mode='sigman')) # ditto remark for theta, will be of shape Sigmay
            dKdtheta = np.asarray(dKdtheta) # should contain Ntheta arrays
            # compute dZdtheta
            Ntheta = theta.size
            dlogp = np.zeros(Ntheta)
            # first get it for sigmaf
            gamma = []
            for i in xrange(D):
                tmp = dKdtheta[0][i].dot(Qs[i])
                gamma.append(np.einsum('ij,ji->i', Qs[i].T, tmp))
            gamma = np.asarray(gamma)
            gamma = kt.kron_diag(gamma)
            kappa = kt.kron_matvec(dKdtheta[0][::-1], alpha)
            dlogp[0] = -self.get_dZdthetai(alpha, kappa, Lambda, gamma)
            # now get it for the length scales
            for i in xrange(1, D+1): # i labels length scales
                # compute the gammas = diag(Qd.T dKddthetai Qd)
                gamma = []
                for j in xrange(D): # j labels dimensions
                    if j == i - 1: # dimension corresponding to l_i is always one less than the index of the length scale
                        tmp = dKdtheta[i].dot(Qs[j]) # this is the dKdtheta corresponding to l_i
                    else:
                        tmp = K[j].dot(Qs[j])
                    gamma.append(np.einsum('ij,ji->i', Qs[j].T, tmp))  # computes only the diagonal of the product
                gamma = np.asarray(gamma)
                gamma = kt.kron_diag(gamma) # exploiting diagonal property of Kronecker product
                dKdtheta_tmp = K.copy()
                dKdtheta_tmp[i-1] = dKdtheta[i]  # can be made more efficient, just set for clarity
                kappa = kt.kron_matvec(dKdtheta_tmp[::-1], alpha)
                dlogp[i] = -self.get_dZdthetai(alpha, kappa, Lambda, gamma)

            # finally get it for sigman
            gamma = []
            for i in xrange(D):
                tmp = dKdtheta[-1][i][:, None]*Qs[i]
                gamma.append(np.einsum('ij,ji->i', Qs[i].T, tmp))
            gamma = np.asarray(gamma)
            gamma = kt.kron_diag(gamma)
            kappa = kt.kron_diag(dKdtheta[-1])*alpha
            dlogp[-1] = -self.get_dZdthetai(alpha, kappa, Lambda, gamma)
            #print theta, logp, dlogp
            #print self.get_full_derivs(K, dKdtheta, alpha, theta[-1]**2*np.eye(self.N), self.yDat)
            return logp, dlogp
        else:
            raise Exception('Mode %s not supported yet' % self.mode)

    def get_dZdthetai(self, alpha, kappa, Lambda, gamma):
        return 0.5*alpha.dot(kappa) - 0.5*sum(gamma/Lambda)

    def get_full_derivs(self, K, dKdtheta, alpha, Lambda, y):
        Kfull = kt.kron_kron(K)
        Kyfull = Kfull + Lambda
        s, logdet = np.linalg.slogdet(Kyfull)
        if s < 0.0:
            raise
        # Lambda, Q = np.linalg.eigh(Kyfull)
        # logdet = np.sum(np.log(Lambda))
        logp = 0.5*y.T.dot(np.linalg.solve(Kyfull, y)) + 0.5*logdet*s + self.N*np.log(2.0*np.pi)/2.0
        dlogp = np.zeros(4)
        dKdthetafull = kt.kron_kron(dKdtheta[0])
        dlogp[0] = 0.5*np.trace(np.linalg.solve(Kyfull, dKdthetafull)) - 0.5*alpha.dot(dKdthetafull.dot(alpha))

        dKdthetafull = K.copy()
        dKdthetafull[0] = dKdtheta[1]
        dKdthetafull = kt.kron_kron(dKdthetafull)
        dlogp[1] = 0.5 * np.trace(np.linalg.solve(Kyfull, dKdthetafull)) - 0.5 * alpha.dot(dKdthetafull.dot(alpha))

        dKdthetafull = K.copy()
        dKdthetafull[1] = dKdtheta[2]
        dKdthetafull = kt.kron_kron(dKdthetafull)
        dlogp[2] = 0.5 * np.trace(np.linalg.solve(Kyfull, dKdthetafull)) - 0.5 * alpha.dot(dKdthetafull.dot(alpha))

        dKdthetafull = dKdtheta[-1]
        dKdthetafull = kt.kron_diag_diag(dKdthetafull)
        dlogp[-1] = 0.5 * np.trace(np.linalg.solve(Kyfull, dKdthetafull)) - 0.5 * alpha.dot(dKdthetafull.dot(alpha))

        return logp, dlogp


class evidence_op(object):
    def __init__(self, y, Ky):
        """
        Computes the (approximate) evidence for GPR
        :param y: the data vector with mean subtracted
        :param Ky: the Ky opereator 
        """
        self.y = y
        self.N = y.size
        self.Ky = Ky

    def logL(self, theta):
        # update hypers
        self.Ky.update_theta(theta)

        # get terms required in marginal likelihood
        alpha = self.Ky.idot(self.y)
        logdet = self.Ky.give_logdet()

        # get marginal likelihood
        logp = (self.y.T.dot(alpha) + logdet + self.N*np.log(2.0*np.pi))/2.0

        # next we get the derivatives (do numerically?)
        # Nhypers = theta.size
        # dlogp = np.zeros(Nhypers)

        print theta, logp
        return logp

