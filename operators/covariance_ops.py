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
        self.Phis = []
        self.Lambdas = []
        for i, k in enumerate(kernels):
            if k == "sqexp":
                self.kernels.append(expsq.sqexp_op(x[i], thetan[i], grid_regular=grid_regular))
                self.Phis.append(self.kernels[i].Phi)
                self.Lambdas.append(self.kernels[i].Lambda)
            else:
                raise Exception("Unsupported kernel %s"%k)
        self.Phis = np.asarray(self.Phis)
        self.Lambdas = np.asarray(self.Lambdas)
        self.Lambdas = kt.kron_diag_diag(self.Lambdas)
        self.PhisT = kt.kron_transpose(self.Phis)
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

    def PhiTdot(self, x):
        return kt.kron_tensorvec(self.PhisT, x)

    def Phidot(self, x):
        return kt.kron_tensorvec(self.Phis, x)

    def set_nugget(self, sigma):
        self.nugget = self.Lambdas + sigma

    def idot(self, x):
        return kt.kron_tensorvec(self.Phis,kt.kron_tensorvec(self.PhisT, x)/self.nugget)

class Ky_op(ssl.LinearOperator):
    def __init__(self, K, Sigmay=None):
        self.K = K  # this is the linear operator representation of the covariance matrix
        if Sigmay is not None:
            # not spherical noise
            self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1]**2*Sigmay))
            # get noise average
            self.eps = np.mean(self.K.theta[-1]**2*Sigmay)
            self.Sigmayinv = np.ones(self.K.kernels[0].N)
            #self.Mop = ssl.aslinearoperator(diags(1.0/(self.K.theta[-1]*np.sqrt(Sigmay))))  # preconditioner suggested by Wilson et. al.
        else:
            # spherical noise
            self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1]**2*np.ones(self.K.N, dtype=np.float64)))
            self.eps = self.K.theta[-1]**2
            self.Mop = ssl.aslinearoperator(diags(1.0 / (self.K.theta[-1] * np.ones(self.K.N, dtype=np.float64))))

        self.shape = (self.K.N, self.K.N)
        self.dtype = np.float64

    def Mop(self):
        return

    def _matvec(self, x):
        return self.K(x) + self.Sigmay(x)

    def _matmat(self, x):
        return self.K._matmat(x) + self.Sigmay(x)

    def idot(self, x):
        tmp = ssl.cg(self.K + self.Sigmay, x, tol=1e-10, M=self.K)
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
    # set inputs
    Nx = 10
    sigmaf = 1.0
    lx = 0.25
    x = np.linspace(-1, 1, Nx)

    Nt = 10
    lt = 1.0
    t = np.linspace(-1, 1, Nt)

    Nz = 10
    lz = 1.5
    z = np.linspace(-1, 1, Nz)

    N = Nx * Nt * Nz
    print "N = ", N

    # draw random vector to multiply with
    print "drawing random vector"
    b = np.random.randn(N)

    # join up all targets
    X = np.array([t, x, z])
    sigman = 0.1
    theta0 = np.array([sigmaf, lt, lx, lz, sigman])

    # instantiate K operator
    print "initialising K"
    Kop = K_op(X, theta0, kernels=["sqexp", "sqexp", "sqexp"], grid_regular=True)

    # set diagonal noise matrix
    print "Setting Sigmay"
    Sigmay = 0.1 * np.ones(N) + np.abs(0.1 * np.random.randn(N))

    # instantiate Ky operator
    print "Initialising Ky"
    Kyop = Ky_op(Kop, Sigmay)

    # test if PhiT Sigmayinv Phi is diagonal
    Phi = kt.kron_kron(Kop.Phis)
    PhiT = kt.kron_kron(Kop.PhisT)
    Lambda = kt.kron_diag_diag(Kop.Lambdas)
    tmp1 = PhiT.dot(Phi)
    tmp2 = Phi.dot(Phi.T)

    Sigmayinv = np.diag(1.0/Sigmay)
    Lambdainv = np.diag(1.0/np.diag(Lambda))
    tmp = Lambdainv + PhiT.dot(Sigmayinv.dot(Phi))

    print np.linalg.cond(tmp)

    print "hello"