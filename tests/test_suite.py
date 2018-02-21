

import numpy as np
from GP.tools import kronecker_tools as kt
from GP.tools import abs_diff
from GP.kernels import exponential_squared
from GP.operators import covariance_ops
import pyfftw


def compare(res1, res2, name):
    print "Diff for %s test = "%name, np.abs(res1-res2).max()

def test_kron_kron(A, K):
    K2 = kt.kron_kron(A)
    compare(K, K2, 'kron_kron')

def test_matvec(A, K, x):
    res1 = np.dot(K, x)
    res2 = kt.kron_matvec(A, x)
    compare(res1, res2, 'matvec')

def test_matmat(A, K, x):
    res1 = np.dot(K, kt.kron_kron(x))
    res2 = kt.kron_matmat(A, x)
    compare(res1, res2, 'matmat')

def test_trace(A, K):
    res1 = np.trace(K)
    res2 = kt.kron_trace(A)
    compare(res1, res2, 'trace')

def test_cholesky(A, K):
    res1 = np.linalg.cholesky(K + 1e-13*np.eye(K.shape[0]))
    L = kt.kron_cholesky(A)
    res2 = kt.kron_kron(L)
    compare(res1, res2, 'cholesky')

def test_cholesky_logdet(A, K):
    s, res1 = np.linalg.slogdet(K + 1e-13*np.eye(K.shape[0]))
    L = kt.kron_cholesky(A)
    res2 = kt.kron_logdet(L)
    compare(s*res1, res2, "cholesky logdet")

def test_eigs_logdet(A, K):
    s, res1 = np.linalg.slogdet(K + 1e-13 * np.eye(K.shape[0]))
    Lambdas, Qs = kt.kron_eig(A)
    Lambda = kt.kron_diag(Lambdas)
    res2 = np.sum(np.log(np.abs(Lambda)))
    compare(res1, res2, 'eigs logdet')

def test_FFT_eigs_det(A, K, sigman):
    # create FFT object
    FFT = pyfftw.builders.fft
    # get eigs by standard method
    N = K.shape[0]
    s, det1 = np.linalg.slogdet(K + sigman**2*np.eye(N))
    D = A.shape[0]
    M = np.zeros(D, dtype=np.int8)
    Lambdas = np.empty(D, dtype=object)
    for d in xrange(D):
        # get first row
        K1 = A[d][0, :]
        # broadcast to Circulant
        Nd = A[d].shape[0]
        M[d] = 2*Nd - 2
        T = np.append(K1, K1[np.arange(Nd)[1:-1][::-1]].conj())
        # get FFT
        xhat = np.sort(FFT(T)().real)[::-1]
        Lambdas[d] = xhat[0:Nd]
    Lambda2 = np.sort(kt.kron_diag(Lambdas))
    M = np.prod(M)
    N = np.prod(N)
    det2 = np.sum(np.log(N*Lambda2/M + sigman**2))
    print det1, det2, np.abs(det1-det2)/np.abs(det1)

def test_eigs(A, K):
    Lambda, Q = np.linalg.eigh(K)
    res1 = np.sort(Lambda)
    Lambdas, Qs = kt.kron_eig(A)
    Lambda2 = kt.kron_diag(Lambdas)
    res2 = np.sort(Lambda2)
    compare(res1, res2, 'eig')

def test_Krec(A, K):
    Lambdas, Qs = kt.kron_eig(A)
    Q = kt.kron_kron(Qs)
    Lambda = kt.kron_diag_diag(Lambdas)
    K2 = Q.dot(np.dot(Lambda, Q.T))
    compare(K, K2, 'Krec')

def test_tensorvec(x, t, z, N):
    Nxp = 10
    xp = np.linspace(-1, 1, Nxp)
    xxp = abs_diff.abs_diff(x, xp)
    Kpx = (kernel.cov_func(thetax, xxp, noise=False)).T

    Ntp = 12
    tp = np.linspace(0, 1, Ntp)
    ttp = abs_diff.abs_diff(t, tp)
    Kpt = (kernel.cov_func(thetat, ttp, noise=False)).T

    Nzp = 14
    zp = np.linspace(-0.5, 1, Nzp)
    zzp = abs_diff.abs_diff(z, zp)
    Kpz = (kernel.cov_func(thetaz, zzp, noise=False)).T

    Ap = np.array([Kpz, Kpx, Kpt])  # note ordering!!!
    Kp = kt.kron_kron(Ap)

    b = np.random.randn(N)
    res1 = Kp.dot(b)
    res2 = kt.kron_tensorvec(Ap, b)
    compare(res1, res2, 'tensorvec')

def test_spherical_noise(A, K):
    N = K.shape[0]
    Sigmay = np.eye(N)
    Ky = K + Sigmay
    Kyinv = np.linalg.inv(Ky)

    # compute eigendecomp with shortened Woodbury matrix identity
    Lambdas, Qs = kt.kron_eig(A)
    Q = kt.kron_kron(Qs)
    Lambda = kt.kron_diag_diag(Lambdas)
    Kyinv2 = Q.dot(np.linalg.inv(Lambda + Sigmay).dot(Q.T))
    compare(Kyinv, Kyinv2, 'spherical noise')


def test_diagonal_noise(A, K):
    N = kt.kron_N(A)
    Sigmay = np.ones(N) + np.abs(np.random.randn(N))
    Ky = K + np.diag(Sigmay)
    Kyinv = np.linalg.inv(Ky)
    Sigmayinv = np.diag(1.0/Sigmay)
    Lambdas, Qs = kt.kron_eig(A)
    Q = kt.kron_kron(Qs)
    Lambda = kt.kron_diag(Lambdas)
    Kyinv2 = Sigmayinv - Sigmayinv.dot(Q.dot(np.linalg.inv(np.diag(1.0/Lambda) + Q.T.dot(Sigmayinv.dot(Q))).dot(Q.T.dot(Sigmayinv))))
    compare(Kyinv, Kyinv2, 'diagonal noise')

def test_Kop_matvec(Kop, K, x):
    res1 = Kop(x)
    res2 = K.dot(x)
    compare(res1, res2, 'Kop matvec')

def test_Kyop_matvec(Kyop, K, Sigmay, x):
    Ky = K + np.diag(Sigmay)
    res1 = Ky.dot(x)
    res2 = Kyop(x)
    compare(res1, res2, 'Kyop matvec')

def test_transpose(A, K):
    AT = kt.kron_transpose(A)
    res1 = kt.kron_kron(AT)
    res2 = K.T
    compare(res1, res2, "Transpose")

def test_Kyopinv(Kyop, K, Sigmay, b):
    Ky = K + np.diag(Sigmay)
    res1 = np.linalg.solve(Ky, b)
    res2 = Kyop.idot(b)
    compare(res1, res2, "Kyopinv")



if __name__=="__main__":
    kernel = exponential_squared.sqexp()
    Nx = 17
    sigmaf = 1.0
    lx = 0.5
    thetax = np.array([sigmaf, lx])
    x = np.linspace(-1, 1, Nx)
    xx = abs_diff.abs_diff(x, x)
    Kx = kernel.cov_func(thetax, xx, noise=False)

    Nt = 19
    lt = 0.5
    thetat = np.array([sigmaf, lt])
    t = np.linspace(-1, 1, Nt)
    tt = abs_diff.abs_diff(t, t)
    Kt = kernel.cov_func(thetat, tt, noise=False)

    Nz = 21
    lz = 0.5
    thetaz = np.array([sigmaf, lz])
    z = np.linspace(-1, 1, Nz)
    zz = abs_diff.abs_diff(z, z)
    Kz = kernel.cov_func(thetaz, zz, noise=False)

    N = Nx * Nt * Nz
    print "N = ", N
    #b = np.random.randn(N)
    b = np.ones(N)

    K = np.kron(Kz, np.kron(Kx, Kt))
    A = np.array([Kz, Kx, Kt])  # note ordering!!!

    if False:
        test_kron_kron(A, K)

    if False:
        test_matvec(A, K, b)

    if False:
        B = np.array([Kz + np.random.randn(Nz, Nz), Kx + np.random.randn(Nx, Nx), Kt + np.random.randn(Nt, Nt)])
        test_matmat(A, K, B)

    if False:
        test_trace(A, K)

    if False:
        test_cholesky(A, K)  # expected to perform poorly since jitter is added differently

    if False:
        test_cholesky_logdet(A, K)

    if False:
        test_eigs_logdet(A, K)

    if False:
        test_eigs(A, K)

    if False:
        test_Krec(A, K)

    if False:
        test_tensorvec(x, t, z, N)

    if False:
        test_spherical_noise(A, K)

    if False:
        test_diagonal_noise(A, K)

    if False:
        test_transpose(A, K)


    X = np.array([z, x, t])
    sigman = 0.1
    theta0 = np.array([sigmaf, lz, lx, lt, sigman])
    Kop = covariance_ops.K_op(X, theta0, kernels=["sqexp", "sqexp", "sqexp"])

    if False:
        test_Kop_matvec(Kop, K, b)

    Sigmay = np.ones(N) + np.abs(np.random.randn(N))

    # nugget = sigman**2*np.mean(Sigmay)
    #
    # Kapprox = Phis.dot(np.diag(Lambdas).dot(PhisT)) + nugget * np.eye(N)
    #
    # Kop.set_nugget(nugget)
    #
    # res1 = Kapprox.dot(b)
    #
    # res2 = Kop.dot2(b)
    #
    # print "Kapprox diff = ", np.abs(res1 - res2).max()
    #
    # Lambda = Kop.Lambdas
    #
    # nuggetinv = np.diag(1.0/(Lambdas + nugget))
    #
    # Kapproxinv = Phis.dot(nuggetinv.dot(PhisT))
    #
    # res1 = Kop.idot(b)
    #
    # res2 = Kapproxinv.dot(b)
    #
    # print "Kapproxinv diff = ", np.abs(res1 - res2).max()
    #
    # res1 = Kapprox.dot(b)
    #
    # #res2 = np.linalg.solve(Kapprox, res1)
    # res2 = Kapproxinv.dot(res1)
    #
    # print "KKinv diff = ", np.abs(b - res2).max()
    #
    # Kexact = K + nugget * np.eye(N)
    # print "K cond = ", np.linalg.cond(Kexact)
    # print "KinvK cond =", np.linalg.cond(Kapproxinv.dot(Kexact))


    Kyop = covariance_ops.Ky_op(Kop, Sigmay)

    if False:
        test_Kyop_matvec(Kyop, K, sigman**2 * Sigmay, b)

    if False:
        test_Kyopinv(Kyop, K, sigman**2 * Sigmay, b)

    if False:
        test_FFT_eigs_det(A, K, sigman)

    # test determinant from eigenvalue approximation
    N2 = 256*2*2
    print N2
    l = 0.01
    theta = np.array([sigmaf, l, sigman])
    x = np.linspace(-1, 1, N2)
    xx = abs_diff.abs_diff(x, x)
    K = kernel.cov_func(theta, xx, noise=True)

    Sigmay = sigman**2*np.ones(N2) #+ 0.5*np.abs(np.random.randn(N2)))

    eps = np.mean(Sigmay)

    Ky = K + np.diag(Sigmay)

    s, det1 = np.linalg.slogdet(Ky)

    # get the first row
    K1 = K[0,:]
    M = 2*N2 - 2
    C = np.append(K1, K1[np.arange(N2)[1:-1][::-1]].conj())
    FFT = pyfftw.builders.fft
    Chat = np.sort(FFT(C)().real)[::-1][0:N2]

    # get approximate determinant from pspec
    det2 = np.sum(np.log(N2*Chat/M + eps))

    print det1, det2, np.abs(det1 - det2)/np.abs(det1)

    # import matplotlib.pyplot as plt
    # plt.figure('pspec')
    # plt.plot(Chat, 'x')
    # plt.show()