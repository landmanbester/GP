#!/usr/bin/env python
"""
Created on Tue Jan 31 16:48:12 2017

@author: landman

2D implementation of sparse GP

"""

import numpy as np
from scipy.special import gamma
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

        # Set bounds for hypers (they must be strictly positive)
        lmin = self.L/(2*self.m) + 1e-5
        self.bnds = ((1e-5, 1.0e2), (lmin, self.L), (1.0e-4, None))

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
            #print "Warning flag raised", thetap[2]
            #print thetap[0]
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

def eigenvals(j, L):
    """
    j is a vector containing integer values 
    """
    return np.sum((np.pi*j/(2.0*L))**2)

def eigenfuncs(j, x, L):
    D = x.shape[0]
    tmp = 1.0
    for i in xrange(D):
        tmp *= np.sin(np.pi*j[i]*(x[i,:] + L[i])/(2.0*L[i]))/np.sqrt(L[i])
    return tmp

def cov_func(x):
    return np.exp(-x**2/2)

def spectral_density(s,D):
    return (2*np.pi)**(D/2)*np.exp(-s**2/2)

def ij_diff(x,xp,mode="same",grid=True):
    """
    Computes the absolute differences | x[i] - xp[j] |.
    Input:
        x = a 2D 2xN vector
        xp = another 2D 2xNp vector
        mode = if mode is same then x = xp and we can save some computation
    """
    if grid:
        # Create grid and flatten
        l,m = np.meshgrid(x[0,:],x[1,:])
        tmp = np.vstack((l.flatten(order='C'),m.flatten(order='C')))
        del l, m #To save memory

        # Get number of data points
        N = x.shape[1]**2
    else:
        tmp = x
        N = x.shape[1]
    
    # abs differences
    XX = tmp[0,:] + 1j*tmp[1,:] #complex trick to get vectorised differences in 2D
    #XX = x[0,:] + 1j*x[1,:]
    if mode=="same":
        XX = np.abs(np.tile(XX,(N,1)).T - np.tile(XX,(N,1)))
        return XX
    else:
        #Create grid and flatten
        #l,m = np.meshgrid(xp[0,:],xp[1,:])
        #tmp = np.vstack((l.flatten(order='C'),m.flatten(order='C')))
        #del l, m #To save memory
        Np = xp.shape[1]
        
        XXp = xp[0,:] + 1j*xp[1,:] #complex trick to get vectorised differences in 2D
        XXp = np.abs(np.tile(XX,(Np,1)).T - np.tile(XXp,(N,1)))
        return XXp

def func2(x,y):
    return 5*np.exp(-(x**2 + y**2)/(2*3))*np.sin(np.sqrt(x**2 + y**2))
    
def func(x,y):
    return 5*np.exp(-(x**2 + y**2)/(2*3))
    
if __name__=="__main__":
    D = 2 # Dimension
    m = 40 # Number of basis funcs
    L = 10 # Domain boundary
    LL = np.tile(L,(1,D)).squeeze()
    N = 100000 # Number of inputs
    x = -5.0 + 10*np.random.random([D,N])
    
    Np = 50
    xp = np.linspace(-5,5,Np)
    Xp, Yp = np.meshgrid(xp,xp)
    xp = np.vstack((Xp.flatten(),Yp.flatten()))
    
    #Get function value with some noise added
    z = func2(x[0,:],x[1,:]) + 0.05*np.random.randn(N)    
    
    zp_exact = func2(Xp,Yp)
    
    j1 = np.kron(np.arange(m).reshape((m,1)),np.ones([m,1]))
    j2 = np.tile(np.arange(m).reshape((m,1)),(m,1))
    j = np.hstack((j1,j2))
    
    # Get the eigenvalues and eigenvectors
    Eigvals = np.zeros(m**D)
    Phi = np.zeros([N,m**D])
    S = np.zeros(m**D)
    for i in xrange(m**D):
        Eigvals[i] = eigenvals(j[i], L)
        Phi[:,i] = eigenfuncs(j[i], x, LL)
        S[i] = spectral_density(np.sqrt(Eigvals[i]),D)
    
    Lambda = np.diag(S)

    # Instantiate GP object
    RR_GP = RR_2DGP(x, xp, L, m, D)

    theta0 = np.array([1.0,1.0,0.05])
    
    zp_coeffs, theta = RR_GP.RR_EvalGP(theta0,z) #[0].reshape((Np**2,Np**2))
    print theta
    
    zp = RR_GP.RR_From_Coeffs(zp_coeffs).reshape((Np,Np))

    #Xp,Yp = np.meshgrid(xp[0,:],xp[1,:])
    
    #Plot reconstructed function
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    
    surf3 = ax3.plot_surface(Xp, Yp, zp, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,alpha=0.5)
    ax3.set_zlim(zp.min(), zp.max())
    
    ax3.zaxis.set_major_locator(LinearLocator(10))
    ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    
    fig3.colorbar(surf3, shrink=0.5, aspect=5)
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection='3d')
    
    surf4 = ax4.plot_surface(Xp, Yp, zp_exact, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,alpha=0.5)
    ax4.set_zlim(zp_exact.min(), zp_exact.max())
    
    ax4.zaxis.set_major_locator(LinearLocator(10))
    ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    fig4.colorbar(surf4, shrink=0.5, aspect=5)
    #ax3.scatter(x[0,:],x[1,:],z,'k')        
##%%
#ra_s, dec_s, I_s, q, u, v, emaj_d, emin_d, pa_d, alphas, freq0 = np.loadtxt("/home/landman/Projects/Data/ddfacet_test_data/sky_models/simple_spi.txt",unpack=True,usecols=(1,2,3,4,5,6,7,8,9,10,11))
#Nsource = 50
#with open("/home/landman/Projects/Data/ddfacet_test_data/sky_models/alpha_map.txt","w") as fp:
#    fp.write("#format:name ra_d dec_d i q u v emaj_d emin_d pa_d spi freq0 \n")
#    for i in xrange(Nsource):
#        fp.write("SRC"+str(i)+" %s %s %s %s %s %s %s %s %s %s %s \n"%(ra_s[i],dec_s[i],alphas[i],0,0,0,0,0,0,alphas[i],1.495e9))
##%%    
##    # Compare covariance functions
##    K_approx = np.dot(Phi,np.dot(Lambda,Phi.T))
##    K = cov_func(XX)
##%%
## Create a pulse train
#N = 200
#
#
## Create a convolution kernel
#x = np.linspace(-2*np.pi,2*np.pi, N)
#y = 1.0 + np.sin(x)
#
#pulse = np.zeros(N)
#pulse[np.arange(0,N,N/10)] = 1.0 + np.sin(x[np.arange(0,N,N/10)])
#
#plt.figure('x',figsize=(15,6))
#plt.stem(x, pulse)
##plt.ylim(0,1.2)


#%%            