"""
Here we build the covariance opertors for GPR
"""


import numpy as np
from scipy.sparse import linalg as ssl
from scipy.sparse import diags


class Ky(ssl.LinearOperator):
    def __init__(self, K, Sigmay=None):
        self.K = K  # this is the linear operator representation of the covariance matrix
        if Sigmay is not None:
            # not spherical noise
            self.Sigmay = diags(self.K.theta[-1]**2*Sigmay)
            self.Mop = diags(1.0/(self.K.theta[-1]*np.sqrt(Sigmay)))  # preconditioner suggested by Wilson et. al.
        else:
            # spherical noise
            self.Sigmay = diags(self.K.theta[-1]**2*np.ones(self.K.N, dtype=np.float64))
            self.Mop = diags(1.0 / (self.K.theta[-1] * np.ones(self.K.N, dtype=np.float64)))


    def _matvec(self, x):
        return self.K(x) + self.Sigmay(x)

    def _matmat(self, x):
        return self.K._matmat(x) + self.Sigmay(x)

    def idot(self, x):
        tmp = ssl.cg(self.K + self.Sigmay, x, tol=1e-10, M=self.Mop)
        if tmp[1] > 0:
            print "Warning cg tol not achieved"
        return tmp[0]

    def dKdtheta(self, theta, K, mode):
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