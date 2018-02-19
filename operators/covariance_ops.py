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
    def __init__(self, x, theta, kernels=["sqexp"]):
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
        # self.Phis = []
        # self.Lambdas = []
        for i, k in enumerate(kernels):
            if k == "sqexp":
                self.kernels.append(expsq.sqexp_op(x[i], thetan[i]))
                # self.Phis.append(self.kernels[i].Phi)
                # self.Lambdas.append(self.kernels[i].S)  # approximate power spectrum
            else:
                raise Exception("Unsupported kernel %s"%k)
        # self.Phis = np.asarray(self.Phis)
        # self.Lambdas = np.asarray(self.Lambdas)
        # self.Lambdas = kt.kron_diag(self.Lambdas)
        # self.PhisT = kt.kron_transpose(self.Phis)
        self.shape = (self.N, self.N)
        self.dtype = np.float64

    def update_theta(self, theta):
        self.theta = theta
        thetan = np.zeros([self.D, 3])
        thetan[:, 0] = theta[0]
        thetan[:, -1] = theta[-1]
        for i in xrange(self.D):  # this is how many length scales will be involved in te problem
            thetan[i, 1] = theta[i + 1]  # assuming we set up the theta vector as [[sigmaf, l_1, sigman], [sigmaf, l_2, ..., sigman]]
            self.kernels[i].update_theta(thetan[i, :])

    def _matvec(self, x):
        return kt.kron_toep_matvec(self.kernels, x, self.N, self.D)

    def _matmat(self, x):
        return kt.kron_toep_matmat(self.kernels, x, self.N, self.D)

    def give_FFT_eigs(self):
        self.FFT_eigs = np.empty(self.D, dtype=object)
        self.Ns = np.zeros(self.D, dtype=np.int8)
        self.Ms = np.zeros(self.D, dtype=np.int8)
        for d in xrange(self.D):
            # get Chat
            Nd = self.kernels[d].N
            self.Ns[d] = Nd
            self.Ms[d] = 2*Nd - 2
            self.FFT_eigs[d] = np.sort(self.kernels[d].Chat.real)[::-1][0:Nd]
        self.FFT_eigs = kt.kron_diag(self.FFT_eigs)
        return self.FFT_eigs, np.prod(self.Ns), np.prod(self.Ms)

    # # still figuring it out below this line!!!!
    # def set_nugget(self, sigma):
    #     self.sigma = sigma
    #     self.nugget = self.Lambdas + sigma
    #     print "Helo"
    #
    # def dot2(self, x):
    #     x = kt.kron_tensorvec(self.PhisT, x)
    #     x *= self.Lambdas
    #     x = kt.kron_tensorvec(self.Phis, x)
    #     x += self.sigma
    #     return x
    #
    # def idot(self, x):
    #     x = kt.kron_tensorvec(self.PhisT, x)
    #     x /= self.nugget
    #     return kt.kron_tensorvec(self.Phis, x)


class Ky_op(ssl.LinearOperator):
    def __init__(self, K, Sigmay=None):
        self.K = K  # this is the linear operator representation of the covariance matrix
        if Sigmay is not None:
            self.diag_noise = Sigmay
            # not spherical noise
            self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1]**2*self.diag_noise))
            # get noise average
            self.eps = np.mean(self.K.theta[-1]**2*Sigmay)
        else:
            self.diag_noise = np.ones(self.K.N, dtype=np.float64)
            # spherical noise
            self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1]**2*self.diag_noise))
            self.eps = self.K.theta[-1]**2

        self.shape = (self.K.N, self.K.N)
        self.dtype = np.float64

    def update_theta(self, theta):
        self.K.update_theta(theta)
        self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1] ** 2 * self.diag_noise))
        self.eps = np.mean(self.K.theta[-1] ** 2 * self.diag_noise)

    def _matvec(self, x):
        return self.K(x) + self.Sigmay(x)

    def _matmat(self, x):
        return self.K._matmat(x) + self.Sigmay(x)

    def idot(self, x):
        tmp = ssl.cg(self.K + self.Sigmay, x, tol=1e-10)
        if tmp[1] > 0:
            print "Warning cg tol not achieved"
        return tmp[0]

    def give_logdet(self):
        eigs, N, M = self.K.give_FFT_eigs()
        # using approximate nugget term eps if Sigmay not specified
        self.logdet = np.sum(np.log(N*eigs/M + self.eps))
        return self.logdet

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