#!/usr/bin/env python
"""
Created on Tue Jan 31 16:48:12 2017

@author: landman

2D implementation of sparse GP

"""

import numpy as np
from scipy.special import gamma
from scipy.signal import fftconvolve
import scipy.optimize as opt
import matplotlib.pyplot as plt
import Class2DGP
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class RR_2DGP(Class2DGP.Class2DGP):
    def __init__(self, x, xp, L, m, D, covariance_function='sqexp'):
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
        Class2DGP.Class2DGP.__init__(self, x, xp, covariance_function=covariance_function)

        # Set the covariance function on spectral density
        self.set_spectral_density(covariance_function=covariance_function)

        self.L = L #args[2]
        LL = np.tile(L,(1,D)).squeeze()
        self.m = m #args[3]
        j1 = np.kron(np.arange(1,m+1).reshape((m,1)),np.ones([m,1]))
        j2 = np.tile(np.arange(1,m+1).reshape((m,1)),(m,1))
        j = np.hstack((j1,j2))

        self.D = D
        
        # Calculate basis functions and eigenvals
        self.Eigvals = np.zeros(self.m**self.D)
        self.Phi = np.zeros([self.N, self.m**self.D])
        self.Phip = np.zeros([self.Np, self.m**self.D])
        for i in xrange(m**D):
            self.Eigvals[i] = self.eigenvals(j[i], L)
            self.Phi[:,i] = self.eigenfuncs(j[i], x, LL)
            self.Phip[:,i] = self.eigenfuncs(j[i], xp, LL)
            #self.S[i] = self.spectral_density(theta0, np.sqrt(Eigvals[i]))
            
        # Compute dot(Phi.T,Phi) # This doesn't change, doesn't depend on theta
        self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
        self.s = np.sqrt(self.Eigvals)
        #print self.s

        # Set bounds for hypers (they must be strictly positive)
        #lmin = self.L/(2*self.m) + 1e-5
        self.bnds = ((1e-5, 1.0e2), (1e-5, 0.9*self.L), (1.0e-4, None))

    def set_spectral_density(self, covariance_function='sqexp'):
        if covariance_function == "sqexp":
            self.spectral_density = lambda theta : self.spectral_density_sqexp(theta, s=self.s)
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
        logQ = (self.N - self.m**2) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
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
        dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.m**2) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
        dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, np.dot(Lambdainv,ZinvPhiTy)) - yTQinvy)/theta[2]

        logp = (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
        dlogp = (dlogQdtheta + dyTQinvydtheta)/2
        return logp, dlogp

    def RR_logp_and_gradlogp_conv(self, theta, y):
        S = self.spectral_density(theta)
        #print S
        Lambdainv = np.diag(1.0 / S)
        Z = self.PhiTPhiconv + theta[2] ** 2 * Lambdainv
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
        logQ = (self.N - self.m**2) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
        # Get the quadratic term
        PhiTy = np.dot(self.Phiconv0.T, y)
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
        dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.m**2) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
        dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, np.dot(Lambdainv,ZinvPhiTy)) - yTQinvy)/theta[2]

        logp = (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
        dlogp = (dlogQdtheta + dyTQinvydtheta)/2
        return logp, dlogp

    def RR_logp_and_gradlogp_conv_MEM(self, theta, y):
        S = self.spectral_density(theta)
        Lambdainv = np.diag(1.0 / S)
        Z = self.PhiTPhiconv + theta[2] ** 2 * Lambdainv
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
        logQ = (self.N - self.m**2) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
        # Get the quadratic term
        PhiTy = np.dot(self.Phiconv0.T, y)
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
        dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.m**2) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
        dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, np.dot(Lambdainv,ZinvPhiTy)) - yTQinvy)/theta[2]

        logp = yTQinvy/2.0 # (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
        dlogp = dyTQinvydtheta/2.0 # (dlogQdtheta + dyTQinvydtheta)/2
        return logp, dlogp

    def eigenvals(self, j, L):
        return np.sum((np.pi*j/(2.0*L))**2)

    def eigenfuncs(self, j, x, L):
        tmp = 1.0
        for i in xrange(self.D):
            tmp *= np.sin(np.pi*j[i]*(x[i,:] + L[i])/(2.0*L[i]))/np.sqrt(L[i])
        return tmp

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
        return theta[0]**2.0*np.sqrt(2.0*np.pi*theta[1]**2)**(self.D/2)*np.exp(-theta[1]**2*s**2/2)

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

    def RR_Give_Coeffs_conv(self, theta, y):
        S = self.spectral_density(theta)
        Z = self.PhiTPhiconv + theta[2] ** 2 * np.diag(1.0 / S)
        try:
            L = np.linalg.cholesky(Z)
        except:
            print "Had to add jitter. Theta = ", theta
            L = np.linalg.cholesky(Z + 1.0e-6*np.eye(Z.shape[0]))
        Linv = np.linalg.inv(L)
        fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phiconv0.T, y)))
        return fcoeffs

    def RR_Reset_Targets(self, xp):
        self.xp = xp
        self.Phip = np.zeros([xp.size, self.m])
        for j in xrange(self.m):
            self.Phip[:, j] = self.eigenfuncs(j, xp)

    def RR_convolve_basis(self, PSF, xx, yy, L):
        """
        Convolve basis funcs with psf.
        Inputs:
            PSF: Np x Np array
            xx: Np x Np gridded x coord
            yy: Np x Np gridded y coord
        """
        j1 = np.kron(np.arange(1,self.m+1).reshape((self.m,1)),np.ones([self.m,1]))
        j2 = np.tile(np.arange(1,self.m+1).reshape((self.m,1)),(self.m,1))
        j = np.hstack((j1,j2))
        LL = np.tile(L, (1, self.D)).squeeze()
        Np = xx.shape[0]
        xy = np.vstack((xx.flatten(), yy.flatten()))
        #self.Phiconv = np.zeros([(2*Np - 1)**2, self.m**2])
        self.Phiconv0 = np.zeros([Np ** 2, self.m ** 2])
        for i in xrange(self.m**2):
            tmp = self.eigenfuncs(j[i], xy, LL)
            #tmp2 = fftconvolve(tmp.reshape(Np,Np), PSF)
            #self.Phiconv[:, i] = tmp2.flatten()
            #self.Phiconv0[:, i] = tmp2[(Np-1)//2:3*Np//2, (Np-1)//2:3*Np//2].flatten()
            self.Phiconv0[:, i] = fftconvolve(tmp.reshape(Np, Np), PSF, mode='same').flatten()
        #self.PhiTPhiconv = np.dot(self.Phiconv.T, self.Phiconv)
        self.PhiTPhiconv = np.dot(self.Phiconv0.T, self.Phiconv0)

    def RR_From_Coeffs(self, coeffs):
        return np.dot(self.Phip, coeffs)

    def RR_From_Coeffs_Degrid(self, coeffs):
        return np.dot(self.Phip_Degrid, coeffs)

    def RR_From_Coeffs_Degrid_ref(self, coeffs):
        Phip = np.zeros(self.m)
        for j in xrange(self.m):
            Phip[j] = self.eigenfuncs(j, 1.0)
        return np.sum(Phip*coeffs)

    def RR_covf(self, theta, return_covf=True):
        S = self.spectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        fcovcoeffs = theta[2] ** 2 * np.dot(Linv.T, Linv)
        if return_covf:
            covf = theta[2] ** 2 * np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, self.Phip.T)))
            return covf, fcovcoeffs
        else:
            return fcovcoeffs

    def RR_covf_conv(self, theta, return_covf=True):
        S = self.spectral_density(theta)
        Z = self.PhiTPhiconv + theta[2] ** 2 * np.diag(1.0 / S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        fcovcoeffs = theta[2] ** 2 * np.dot(Linv.T, Linv)
        if return_covf:
            covf = theta[2] ** 2 * np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, self.Phip.T)))
            return covf, fcovcoeffs
        else:
            return fcovcoeffs

    def RR_trainGP(self, theta0, y):
        # Do optimisation
        self.SolverFlag = 0
        thetap = opt.fmin_l_bfgs_b(self.RR_logp_and_gradlogp, theta0, fprime=None, args=(y,), bounds=self.bnds) #, factr=1e10, pgtol=0.1)

        if np.any(np.isnan(thetap[0])):
            raise Exception('Solver crashed error. Are you trying a noise free simulation? Use FreqMode = Poly instead.')

        #Check for convergence
        if thetap[2]["warnflag"]:
            self.SolverFlag = 1
            #print "Warning flag raised", thetap[2]
            #print thetap[0]
        # Return optimised value of theta
        return thetap[0]

    def RR_trainGP_conv(self, theta0, y):
        # Do optimisation
        self.SolverFlag = 0
        thetap = opt.fmin_l_bfgs_b(self.RR_logp_and_gradlogp_conv, theta0, fprime=None, args=(y,),
                                   bounds=self.bnds)  # , factr=1e10, pgtol=0.1)

        if np.any(np.isnan(thetap[0])):
            raise Exception(
                'Solver crashed error. Are you trying a noise free simulation? Use FreqMode = Poly instead.')

        # Check for convergence
        if thetap[2]["warnflag"]:
            self.SolverFlag = 1
            # print "Warning flag raised", thetap[2]
            # print thetap[0]
        # Return optimised value of theta
        return thetap[0]

    def RR_trainGP_conv_MEM(self, theta0, y):
        # Do optimisation
        self.SolverFlag = 0
        thetap = opt.fmin_l_bfgs_b(self.RR_logp_and_gradlogp_conv_MEM, theta0, fprime=None, args=(y,),
                                   bounds=self.bnds)  # , factr=1e10, pgtol=0.1)

        if np.any(np.isnan(thetap[0])):
            raise Exception(
                'Solver crashed error.')

        # Check for convergence
        if thetap[2]["warnflag"]:
            self.SolverFlag = 1
            # print "Warning flag raised", thetap[2]
            # print thetap[0]
        # Return optimised value of theta
        return thetap[0]

    def RR_EvalGP_conv_MEM(self, theta0, y):
        theta = self.RR_trainGP_conv_MEM(theta0, y)
        coeffs = self.RR_Give_Coeffs_conv(theta, y)
        return coeffs, theta

    def RR_EvalGP_conv(self, theta0, y):
        theta = self.RR_trainGP_conv(theta0, y)
        coeffs = self.RR_Give_Coeffs_conv(theta, y)
        return coeffs, theta

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