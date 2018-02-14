
"""
Some tools to exploit fast matrix vector products using the FFT. Inputs need to be on a regular grid. 
"""

import numpy as np
import nifty2go as ift

def FFT_circvec(c, x):
    """
    Computes matrix vector product y = Cx where C = circulant(c) in NlogN time
    :param c: the upper row vector of circulant matrix C 
    :param x: the vector to multiply with
    :return: y the result
    """
    Lambda = np.fft.rfft(c)
    xk = np.fft.rfft(x)
    return np.fft.irfft(Lambda*xk)

def FFT_circmat(c, x):
    """
    Computes matrix matrix product y = CX where C = circulant(c) in NlogN time
    :param c: the upper row vector of circulant matrix C 
    :param x: the matrix to multiply with
    :return: y the result
    """
    Lambda = np.fft.rfft(c)
    y = np.zeros([c.size, x.shape[1]])
    for i in xrange(x.shape[1]):
        xk = np.fft.rfft(x)
        y[:, i] = np.fft.irfft(Lambda*xk)
    return y

def FFT_toepvec(t, x):
    """
    Computes matrix vector product y = Kx where K is a covariance (i.e. Hermitian Toeplitz) matrix. 
    Here t is the first row/column of the covariance matrix 
    :param t: the upper row vector of the covariance matrix 
    :param x: the vector to multiply with
    :return: y the result
    """
    # broadcast to circulant
    N = t.size
    c = np.append(t, t[np.arange(N)[1:-1][::-1]].conj())
    x_aug = np.append(x, np.zeros(N-2))
    tmp = FFT_circvec(c, x_aug)
    return tmp[0:x.size]

def FFT_toepmat(t, x):
    """
    Computes matrix matrix product y = Kx where K is a covariance (i.e. Hermitian Toeplitz) matrix. 
    Here t is the first row/column of the covariance matrix 
    :param t: the upper row vector of the covariance matrix 
    :param x: the matrix to multiply with
    :return: y the result
    """
    # broadcast to circulant
    N, M = x.shape
    c = np.append(t, t[np.arange(N)[1:-1][::-1]].conj())
    x_aug = np.append(x, np.zeros([N-2, M]))
    tmp = FFT_circmat(c, x_aug)
    return tmp[0:x.size, :]

if __name__=="__main__":
    from GP.tools import abs_diff
    from GP.kernels import exponential_squared

    # get Toeplitz matrix
    kernel = exponential_squared.sqexp()
    Nx = 1024
    thetax = np.array([1.0, 0.5])
    x = np.linspace(-10, 10, Nx)
    xx = abs_diff.abs_diff(x, x)
    Kx = kernel.cov_func(thetax, xx, noise=False)

    # broadcast to circulant row
    row1 = np.append(Kx[0, :], Kx[0, np.arange(Nx)[1:-1][::-1]].conj())

    # s_space = ift.RGSpace([Nx, Nx])
    # FFT = ift.FFTOperator(s_space)
    # h_space = FFT.target[0]
    # binbounds = ift.PowerSpace.useful_binbounds(h_space, logarithmic=False)
    # p_space = ift.PowerSpace(h_space, binbounds=binbounds)
    # s_spec = (lambda s: np.sqrt(2*np.pi)*thetax[0]**2.0*thetax[1]*np.exp(-thetax[1]**2*s**2/2.0))
    # S = ift.create_power_operator(h_space, power_spectrum=s_spec)
    # Projection = ift.PowerProjectionOperator(domain=h_space, power_space=p_space)



    # compare eigenvalues to fft coefficients
    import scipy.linalg as scl

    # Kcirc = scl.circulant(row1)
    # Lambda, Q = np.linalg.eigh(Kcirc)

    Lambda2 = np.fft.fft(row1).real

    #print "Eig diff = ", np.abs(np.sort(Lambda) - np.sort(Lambda2)).max(), np.abs(Lambda - Lambda2).min()

    # get eigenvalues of Kx
    Lambda3, Q3 = np.linalg.eigh(Kx)

    Lambda4 = np.sort(Lambda2)[Nx-2::]

    print "Eig diff 2 = ", np.abs(np.sort(Lambda3) - np.sort(Lambda4)).max(), np.abs(Lambda3 - Lambda4).min()

    det1 = np.sum(np.log(np.abs(Lambda3)))
    det2 = np.sum(np.log(np.abs(Lambda4)))
    #
    # s = np.fft.rfftfreq(Nx, xx[0,1] - xx[0,0])
    #
    # pspec = kernel.spectral_density(thetax, s)
    #
    # s, det4 = np.linalg.slogdet(Kx)


    print "Det diff = ", det1, det2, np.abs(det2-det1)/np.abs(det1)
    # t = np.random.randn(row1.size)
    # res1 = Kcirc.dot(t)
    # res2 = FFT_circvec(row1, t)
    # print "Circvec diff =", np.abs(res1 - res2).max()
    #
    # # test circvec
    # res1 = np.dot(Kx, t[0:Nx])
    #
    # res2 = FFT_toepvec(Kx[0, :], t[0:Nx])
    # print "Toepvec diff = ", np.abs(res1 - res2).max()

    # import matplotlib.pyplot as plt
    #
    # # plt.figure('Toepvec Diff ')
    # # plt.plot(res1 - res2)
    # plt.figure('Spectra 1')
    # plt.plot(Lambda3, 'x')
    # plt.figure("Spectra 2")
    # plt.plot(np.sqrt(pspec), 'x')
    # # plt.figure("Spectra 3")
    # # plt.plot(Lambda3, 'bx')
    # # plt.plot(Lambda4, 'rx')
    # plt.figure('pspec')
    # plt.plot(pspec, 'x')
    # plt.show()




