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
        self.Phis = np.empty(self.D, dtype=object)
        self.Lambdas = np.empty(self.D, dtype=object)
        for i, k in enumerate(kernels):
            if k == "sqexp":
                self.kernels.append(expsq.sqexp_op(x[i], thetan[i]))
                self.Phis[i] = self.kernels[i].Phi
                self.Lambdas[i] = self.kernels[i].S  # approximate power spectrum
            else:
                raise Exception("Unsupported kernel %s"%k)
        self.PhisT = kt.kron_transpose(self.Phis)
        self.shape = (self.N, self.N)
        self.dtype = np.float64

    def set_preconds(self, eps):
        """
        Sets attributes required for preconditioning operator
        :param eps: this is the average noise level i.e. mean(diag(Sigmay)) 
        """
        for i in xrange(self.D):
            self.Lambdas[i] = self.kernels[i].S
        self.Lambdainv = 1.0 / kt.kron_diag(self.Lambdas)
        self.Sigmainv = 1.0/(np.ones(self.N)*eps)
        self.count = 0
        self.Z = np.diag(self.Lambdainv) + kt.kron_tensormat(self.PhisT, self.Phis) / self.theta[-1] ** 2

    def update_theta(self, theta, eps):
        """
        Updates all attributes depending on theta
        :param theta: vector of hyperparameters 
        :param eps: the average noise level i.e. mean(diag(Sigmay)) 
        :return: 
        """
        self.theta = theta
        thetan = np.zeros([self.D, 3])
        thetan[:, 0] = theta[0]
        thetan[:, -1] = theta[-1]
        for i in xrange(self.D):  # this is how many length scales will be involved in te problem
            thetan[i, 1] = theta[i + 1]  # assuming we set up the theta vector as [[sigmaf, l_1, l_2, ..., sigman]]
            self.kernels[i].update_theta(thetan[i, :])
        self.set_preconds(eps)

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

    # low rank approximation inverse for preconditioning
    def idot(self, x):
        self.count += 1
        Sigmainvx = self.Sigmainv*x
        rhs_vec = kt.kron_tensorvec(self.PhisT, Sigmainvx)
        rhs_vec = np.linalg.solve(self.Z, rhs_vec)
        rhs_vec = kt.kron_tensorvec(self.Phis, rhs_vec)
        return Sigmainvx - self.Sigmainv*rhs_vec


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

        # set mandatory attributes
        self.shape = (self.K.N, self.K.N)
        self.dtype = np.float64

        # set preconditioner
        self.K.set_preconds(self.eps)
        self.Mop = ssl.LinearOperator((self.K.N, self.K.N), matvec=self.K.idot)

    def update_theta(self, theta):
        self.eps = np.mean(theta[-1] ** 2 * self.diag_noise)
        self.K.update_theta(theta, self.eps)
        self.Sigmay = ssl.aslinearoperator(diags(self.K.theta[-1] ** 2 * self.diag_noise))
        self.Mop = ssl.LinearOperator((self.K.N, self.K.N), matvec=self.K.idot)

    def _matvec(self, x):
        return self.K(x) + self.Sigmay(x)

    def _matmat(self, x):
        return self.K._matmat(x) + self.Sigmay(x)

    def idot(self, x):
        tmp = ssl.cg(self.K + self.Sigmay, x, tol=1e-8, M=self.Mop)
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
    Nx = 21
    sigmaf = 1.0
    lx = 0.25
    x = np.linspace(0, 1, Nx)

    Nt = 23
    lt = 0.5
    t = np.linspace(-1, 0, Nt)

    Nz = 25
    lz = 0.35
    z = np.linspace(-2, -1, Nz)

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
    Kop = K_op(X, theta0, kernels=["sqexp", "sqexp", "sqexp"])

    # set diagonal noise matrix
    print "Setting Sigmay"
    Sigmay = np.ones(N) + np.abs(0.2 * np.random.randn(N))

    # Sigmay[500] = 10
    # Sigmay[1500] = 0.001

    # instantiate Ky operator
    print "Initialising Ky"
    Kyop = Ky_op(Kop, Sigmay)

    # test preconditioner
    print "Testing inverse"
    res = Kyop(b)
    import time
    ti = time.time()
    res2 = Kyop.idot(res)
    print "Time taken = ", time.time() - ti
    print np.abs(b - res2).max(), np.abs(b - res2).min()

    # import matplotlib.pyplot as plt
    # plt.figure('Z')
    # plt.imshow(np.log(np.abs(Kop.Z)+1e-15))
    # plt.colorbar()
    # plt.figure("diffs")
    # plt.plot(b, 'kx')
    # plt.plot(res2, 'rx')
    # plt.show()

    print "Niter = ", Kop.count

    # test update theta
    theta = np.array([1.5*sigmaf, 2*lt, 2*lx, 2*lz, 0.1*sigman])
    Kyop.update_theta(theta)

   # test preconditioner
    print "Testing inverse 2"
    res = Kyop(b)
    import time
    ti = time.time()
    res2 = Kyop.idot(res)
    print "Time taken 2 = ", time.time() - ti
    print np.abs(b - res2).max(), np.abs(b - res2).min()

    print "Niter = ", Kop.count

    # test update theta
    theta = np.array([sigmaf, lt, lx, lz, sigman])
    Kyop.update_theta(theta)

   # test preconditioner
    print "Testing inverse 2"
    res = Kyop(b)
    import time
    ti = time.time()
    res2 = Kyop.idot(res)
    print "Time taken 2 = ", time.time() - ti
    print np.abs(b - res2).max(), np.abs(b - res2).min()

    print "Niter = ", Kop.count