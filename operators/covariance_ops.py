"""
Here we build the covariance opertors for GPR
"""


import numpy as np
from GP.tools import abs_diff
from scipy.sparse import linalg as ssl
from scipy.sparse import diags
from GP.tools import kronecker_tools as kt
from GP.kernels import exponential_squared as expsq
from GP.tools import FFT_tools as ft


class K_op(ssl.LinearOperator):
    def __init__(self, x, theta, kernels=["sqexp"], grid_regular=True):
        self.D = x.shape[0]
        self.N = kt.kron_N(x)
        self.x = x
        thetan = np.zeros([self.D, 3])
        thetan[:, 0] = theta[0]
        thetan[:, -1] = theta[-1]
        for i in xrange(self.D):  # this is how many length scales will be involved in te problem
            thetan[i, 1] = theta[i + 1]  # assuming we set up the theta vector as [[sigmaf, l_1, sigman], [sigmaf, l_2, ..., sigman]]
        self.theta = theta
        # set up kernels for each dimension
        self.kernels = []
        for i, k in enumerate(kernels):
            if k == "sqexp":
                self.kernels.append(expsq.sqexp_op(x[i], thetan[i], grid_regular=grid_regular))
            else:
                raise Exception("Unsupported kernel %s"%k)
        self.shape = (self.N, self.N)
        self.dtype = np.float64
        self.grid_regular = grid_regular

    def _matvec(self, x):
        if self.grid_regular:
            return kt.kron_toep_matvec(self.kernels, x, self.N, self.D)
        else:
            return kt.kron_matvec_op(self.kernels, x, self.D)

    def _matmat(self, x):
        return kt.kron_toep_matmat(self.kernels, x, self.N, self.D)

class Ky_op(ssl.LinearOperator):
    def __init__(self, K, Sigmay=None):
        self.K = K  # this is the linear operator representation of the covariance matrix
        if Sigmay is not None:
            # not spherical noise
            self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1]**2*Sigmay))
            self.Mop = ssl.aslinearoperator(diags(1.0/(self.K.theta[-1]*np.sqrt(Sigmay))))  # preconditioner suggested by Wilson et. al.
        else:
            # spherical noise
            self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1]**2*np.ones(self.K.N, dtype=np.float64)))
            self.Mop = ssl.aslinearoperator(diags(1.0 / (self.K.theta[-1] * np.ones(self.K.N, dtype=np.float64))))

        self.shape = (self.K.N, self.K.N)
        self.dtype = np.float64


    def _matvec(self, x):
        return self.K(x) + self.Sigmay(x)

    def _matmat(self, x):
        return self.K._matmat(x) + self.Sigmay(x)

    def idot(self, x):
        tmp = ssl.cg(self.K + self.Sigmay, x, tol=1e-10, M=self.Mop)
        if tmp[1] > 0:
            print "Warning cg tol not achieved"
        return tmp[0]

    def dKdtheta(self, theta, x, K, mode):
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

if __name__=="__main__":
    N = 1024
    x = np.linspace(-10, 10, N)
    sigmaf = 1.0
    l = 1.0
    sigman = 0.1
    theta = np.array([sigmaf, l, sigman])

    # get operator representation
    Kop = esq.sqexp_op(x, theta)

    # get dense representation
    kernel = esq.sqexp()
    xx = abs_diff.abs_diff(x, x)
    Kmat = kernel.cov_func(theta, xx, noise=False)

    # create noise variance
    Sigmay = 0.1 * np.ones(N) + np.abs(0.1 * np.random.randn(N))

    # set Ky operator
    Ky = Ky_op(Kop, Sigmay=Sigmay)

    # test matvec
    t = np.random.randn(N)
    res1 = Ky(t)

    res2 = Kmat.dot(t) + np.diag(sigman**2*Sigmay).dot(t)

    print np.abs(res1 - res2).max()

    # test matmat
    Ky = Ky_op(Kop, Sigmay=Sigmay)

    # test matvec
    tt = np.random.randn(N, 10)
    res1 = Ky._matmat(tt)

    res2 = Kmat.dot(tt) + np.diag(sigman**2*Sigmay).dot(tt)

    print np.abs(res1 - res2).max()

    # test idot
    Kymat = Kmat + np.diag(sigman**2*Sigmay)
    res1 = np.linalg.solve(Kymat, t)

    res2 = Ky.idot(t)

    print np.abs(res1 - res2).max()