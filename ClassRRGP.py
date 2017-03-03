#!/usr/bin/env python

import numpy as np
import scipy as scp
from scipy.linalg import solve_triangular as soltri
from scipy import optimize as opt
import ClassGP
import matplotlib.pyplot as plt
import time


class RR_GP(ClassGP.ClassGP):
    def __init__(self, x, xp, L, m, covariance_function='sqexp'):
        """
        A class to implement reduced rank GPR
        Args:
            x: input (locations where we have data)
            xp: targets (locations where we want to evaluate the function
            L: the boundary of the domain on which we evaluate the eigenfunctions of the Laplacian
            m: the number of basis function to use
            covariance_function: which covariance function to use (currently only squared exponential 'sqexp' supported)
        """
        # Initialise base class
        ClassGP.ClassGP.__init__(self, x, xp, covariance_function=covariance_function)
        self.L = L #args[2]
        self.m = m #args[3]

        # Calculate basis functions and eigenvals
        self.Eigvals = np.zeros(self.m)
        self.Phi = np.zeros([self.N, self.m])
        self.Phip_Degrid = np.zeros([self.Np, self.m])
        for j in xrange(m):
            self.Eigvals[j] = self.eigenvals(j+1)
            self.Phi[:, j] = self.eigenfuncs(j+1, x)
            self.Phip_Degrid[:, j] = self.eigenfuncs(j+1, xp)
        self.Phip = self.Phi
        # Compute dot(Phi.T,Phi) # This doesn't change, doesn't depend on theta
        self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
        self.s = np.sqrt(self.Eigvals)

        # Set the covariance function on spectral density
        self.set_spectral_density(covariance_function=covariance_function)

        # Set bounds for hypers (they must be strictly positive)
#        lmin = self.L/(2*self.m)
#        print lmin
        self.bnds = ((1e-5, None), (1e-6, None), (1.0e-4, None))

    def set_spectral_density(self, covariance_function='sqexp'):
        if covariance_function == "sqexp":
            self.spectral_density = lambda theta : self.spectral_density_sqexp(theta, self.s)
            self.dspectral_density = lambda theta, S, mode: self.dspectral_density_sqexp(theta, S, self.s, mode=mode)
        elif covariance_function == 'mat52':
            self.spectral_density = lambda theta : self.spectral_density_mat52(theta, self.s)
            self.dspectral_density = lambda theta, S, mode: self.dspectral_density_mat52(theta, S, self.s, mode=mode)
        elif covariance_function == 'mat72':
            self.spectral_density = lambda theta: self.spectral_density_mat72(theta, self.s)
            self.dspectral_density = lambda theta, S, mode: self.dspectral_density_mat72(theta, S, self.s, mode=mode)

    def RR_logp_and_gradlogp(self, theta, y):
        S = self.spectral_density(theta)
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
        Zinv = np.dot(Linv.T,Linv)
        logdetZ = 2.0 * np.sum(np.log(np.diag(L)))
        # Get the log term
        logQ = (self.N - self.m) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
        # Get the quadratic term
        PhiTy = np.dot(self.Phi.T, y)
        ZinvPhiTy = np.dot(Zinv, PhiTy)
        yTy = np.dot(y.T, y)
        yTQinvy = (yTy - np.dot(PhiTy.T, ZinvPhiTy)) / theta[2] ** 2
        # Get their derivatives
        dlogQdtheta = np.zeros(theta.size)
        dyTQinvydtheta = np.zeros(theta.size)
        for i in xrange(theta.size-1):
            dSdtheta = self.dspectral_density(theta, S, mode=i)
            dlogQdtheta[i] = np.sum(dSdtheta/S) - theta[2]**2*np.sum(dSdtheta/S*np.diag(Zinv)/S)
            dyTQinvydtheta[i] = -np.dot(ZinvPhiTy.T, np.dot(np.diag(dSdtheta/S **2), ZinvPhiTy))
        # Get derivatives w.r.t. sigma_n
        dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.m) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
        dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, np.dot(Lambdainv,ZinvPhiTy)) - yTQinvy)/theta[2]

        logp = (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
        dlogp = (dlogQdtheta + dyTQinvydtheta)/2
        return logp, dlogp

    def eigenvals(self, j):
        return (np.pi * j / (2.0 * self.L)) ** 2

    def eigenfuncs(self, j, x):
        return np.sin(np.pi * j * (x + self.L) / (2.0 * self.L)) / np.sqrt(self.L)

    def spectral_density_mat52(self, theta, s):  # This is for Mattern with v=5/2
        return theta[0] ** 2 * 2 * 149.07119849998597 / (theta[1] ** 5 * (5.0 / theta[1] ** 2 + s ** 2) ** 3)

    def dspectral_density_mat52(self, theta, S, s, mode=0):
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return S * (-5 / theta[1] + 30 / (theta[1] ** 3 * (5.0 / theta[1] ** 2 + s ** 2)))

    def spectral_density_mat72(self, theta, s):  # This is for Mattern with v=5/2
        return theta[0] ** 2 * 2 * 5807.95327805 / (theta[1] ** 7 * (7.0 / theta[1] ** 2 + s ** 2) ** 3)

    def dspectral_density_mat72(self, theta, S, s, mode=0):
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return S * (-7 / theta[1] + 56 / (theta[1] ** 3 * (7.0 / theta[1] ** 2 + s ** 2)))

    def spectral_density_sqexp(self, theta, s):
        return theta[0]**2.0*np.sqrt(2.0*np.pi*theta[1]**2)*np.exp(-theta[1]**2*s**2/2)

    def dspectral_density_sqexp(self, theta, S, s, mode=0):
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return S/theta[1] - s**2*theta[1]*S
            
    def RR_meanf(self, theta, y):
        S = self.spectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, y)))
        fbar = np.dot(self.Phip, fcoeffs)
        return fbar, fcoeffs

    def RR_Give_Coeffs(self,theta, y):
        S = self.spectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        try:
            L = np.linalg.cholesky(Z)
        except:
            print "Had to add jitter. Theta = ", theta
            L = np.linalg.cholesky(Z + 1.0e-6*np.eye(Z.shape[0]))
        Linv = np.linalg.inv(L)
        fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, y)))
        return fcoeffs

    def RR_Reset_Targets(self, xp):
        self.xp = xp
        self.Phip = np.zeros([xp.size, self.m])
        for j in xrange(self.m):
            self.Phip[:, j] = self.eigenfuncs(j, xp)

    def RR_From_Coeffs(self, coeffs):
        return np.dot(self.Phip, coeffs)

    def RR_From_Coeffs_Degrid(self, coeffs):
        return np.dot(self.Phip_Degrid, coeffs)

    def RR_From_Coeffs_Degrid_ref(self, coeffs):
        Phip = np.zeros(self.m)
        for j in xrange(self.m):
            Phip[j] = self.eigenfuncs(j, 1.0)
        return np.sum(Phip*coeffs)

    def RR_covf(self, theta):
        S = self.dspectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        fcovcoeffs = theta[2] ** 2 * np.dot(Linv.T, Linv)
        covf = theta[2] ** 2 * np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, self.Phip.T)))
        return covf, fcovcoeffs

    def RR_trainGP(self, theta0, y):
        # Do optimisation
        self.SolverFlag = 0
        thetap = opt.fmin_l_bfgs_b(self.RR_logp_and_gradlogp, theta0, fprime=None, args=(y,), bounds=self.bnds) #, factr=1e10, pgtol=0.1)

        if np.any(np.isnan(thetap[0])):
            raise Exception('Solver crashed error. Are you trying a noise free simulation? Use FreqMode = Poly instead.')

        #Check for convergence
        if thetap[2]["warnflag"]:
            self.SolverFlag = 1
            print "Warning flag raised", thetap[2]
            print thetap[0]
        # Return optimised value of theta
        return thetap[0]

    def RR_EvalGP(self, theta0, y):
        theta = self.RR_trainGP(theta0, y)
        coeffs = self.RR_Give_Coeffs(theta, y)
        return coeffs, theta

    def mean_and_cov(self, theta):
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / self.S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        self.fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, self.y)))
        fbar = np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, self.y))))
        self.fcovcoeffs = theta[2] ** 2 * np.dot(Linv.T, Linv)
        covf = theta[2] ** 2 * np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, self.Phip.T)))
        return fbar, covf


if __name__ == "__main__":
    L = 4.0
    m = 50
#    N = 25
#    Np = 100
#    xmax = 2.0
#    x = np.linspace(0.0, xmax, N)
#    sigman = 0.15
#    alpha = 1.5
#    y = x**alpha + sigman * np.random.randn(N)
#    xp = np.linspace(0.1, xmax, Np)
#    yp = xp ** alpha
#    sigmaf0 = (y.max() - y.min())
#    l0 =xmax/2
#    sigman0 = np.var(y)
#    theta = np.array([sigmaf0, l0, sigman0])
#    print theta
#
#    # Instantiate object
#    GP = RR_GP(x, xp, L, m, covariance_function='sqexp')
#
#    # Set the normal GP parameters for comparison
#    GP.set_abs_diff()
#
#    # Train both RR and normal case
#    t1 = time.time()
#    fbar, thetaf = GP.EvalGP(theta, y)
#    t2 = time.time()
#    fcoeffs, RR_thetaf = GP.RR_EvalGP(theta, y)
#    RR_fbar = GP.RR_From_Coeffs_Degrid(fcoeffs)
#    t3 = time.time()
#
#    print t3-t2, t2-t1
#
#    # compare marginals
#    RR_logp, RR_dlogp = GP.RR_logp_and_gradlogp(RR_thetaf, y)
#    logp, dlogp = GP.logp_and_gradlogp(thetaf, y)
#
#    print RR_logp, logp
#    print RR_thetaf, thetaf
##    RR_fbar, coeffs = GP.RR_meanf(theta,y)
##    fbar = GP.meanf(theta, y)
#
#    plt.plot(xp,RR_fbar, 'b')
#    plt.plot(xp, fbar, 'g')
#    plt.plot(xp, yp, 'k')
#    plt.show()

    # fbar = GP.fbar
    # covf = GP.covf
    # fcoeffs = GP.fcoeffs
    # fcovcoeffs = GP.fcovcoeffs
    # fcovcoeffs2 = np.diag(GP.S)
    #
    # logpt = GP.negloglik(theta)

    # # Draw some samples
    # samps = np.random.multivariate_normal(fbar, covf, 100).squeeze().T
    #
    # # Samples some random coeffs
    # fcoeffssamps = np.random.multivariate_normal(fcoeffs, fcovcoeffs, 100).squeeze().T
    # # fcoeffssamps2 = np.random.multivariate_normal(fcoeffs,fcovcoeffs2)
    # samps2 = np.dot(GP.Phip, fcoeffssamps)
    #
    # # Plot results
    # plt.plot(xp, samps, 'g', alpha=0.5)
    # plt.plot(xp, samps2, 'r', alpha=0.5)
    # plt.plot(xp, fbar, 'k')
    #
    # plt.show()


    #    XX = np.tile(x,(N,1)).T - np.tile(x,(N,1))
    #
    #    Ntest = 500
    #    lvals = np.linspace(0.1,10,Ntest)
    #    sigmafvals = np.linspace(5.0,20.0,Ntest)
    #    sigmanvals = np.linspace(0.01,0.5,Ntest)
    #    logpt = np.zeros(Ntest)
    #    logpt2 = np.zeros(Ntest)
    #    for i in xrange(Ntest):
    #        theta = np. array([12.5,lvals[i],0.15])
    #        #theta = np. array([sigmafvals[i],3.4,0.15])
    #        #theta = np. array([13.0,3.4,sigmanvals[i]])
    #        logpt[i] = GP.negloglik(theta)
    #        logpt2[i] = logp(theta,XX,y,N)
    #
    #    plt.figure('x')
    #    plt.plot(lvals,logpt,'b')
    #    plt.plot(lvals,logpt2,'g')

    # Load some data    
    y = np.loadtxt("/home/landman/Projects/Kela/size.txt")
    x = np.loadtxt("/home/landman/Projects/Kela/frequencies.txt")/1.0e9
    N = x.size
    
    # Set flag indices
    I1 = np.arange(80,85)
    I2 = np.arange(546,553)
    I3 = np.arange(569,581)
    I4 = np.arange(611,619)
    I5 = np.arange(638,640)
    I5 = np.arange(671,675)
    I = np.hstack((I1,I2,I3,I4,I5))
    
    # Remove flagged data
    y2 = np.delete(y,I)
    x2 = np.delete(x,I)

    I = np.argsort(x2)
    x2 = x2[I]
    y2 = y2[I]
    
    P = np.polyfit(x2,y2,4)
    
    # Get number of unflagged data points
    N2 = x2.size
    
    # Set targets (same as the points at which we have data)
    Np = N 
    xp = x 

    # Evaluate the polynomial
    fpoly = P[0]*xp**4 + P[1]*xp**3 + P[2]*xp**2 + P[3]*xp + P[4]
    fpoly2 = P[0]*x2**4 + P[1]*x2**3 + P[2]*x2**2 + P[3]*x2 + P[4]
    
    # Instantiate object
    GP = RR_GP(x2, xp, L, m, covariance_function='sqexp')
    
    # Init theta0
    theta0 = np.array([1.0,0.01,0.001])
    
    coeffs, theta = GP.RR_EvalGP(theta0,y2-fpoly2)
    
    print theta
    
    fp = GP.RR_From_Coeffs_Degrid(coeffs)
    
    plt.plot(x2,y2,'rx',label='Data')
    plt.plot(xp,fpoly + fp)