#!/usr/bin/env python
"""
Created on Tue Jan 31 19:42:37 2017

@author: landman
"""

import numpy as np
import scipy as scp
from scipy.linalg import solve_triangular as soltri
from scipy import optimize as opt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Class2DGP(object):
    """
    A class to implement 2D GPR
    """
    def __init__(self, x, xp, covariance_function = 'sqexp'):
        """
        Input:
                x = [N,2] vector of inputs
                xp = [N,2] vector of targets
        """
        self.x = x
        self.xp = xp
        self.N = self.x.shape[1]
        self.Np = self.xp.shape[1]
        self.set_covariance(covariance_function=covariance_function)

    def set_abs_diff(self):
        self.XX = self.abs_diff(self.x,self.x, grid=False)
        self.XXp = self.abs_diff(self.xp, self.x, mode=1).T
        self.XXpp = self.abs_diff(self.xp, self.xp)

    def abs_diff(self,x, xp, mode="same", grid=True):
        """
        Computes the absolute differences | x[i] - xp[j] |.
        Input:
            x = a 2D 2xN vector
            xp = another 2D 2xNp vector
            mode = if mode is same then x = xp and we can save some computation
            grid = whether to evaluate on a grid
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

    def diag_dot(self,A,B):
        """
        Computes the diagonal of C = AB where A and B are square matrices
        """
        D = np.zeros(A.shape[0])
        for i in xrange(A.shape[0]):
            D[i] = np.dot(A[i,:],B[:,i])
        return D

    def set_covariance(self,covariance_function='sqexp'):
        if covariance_function == "sqexp":
            self.cov_func = lambda theta, x, mode : self.cov_func_sqexp(theta, x, mode=mode)
            self.dcov_func = lambda theta, x, mode : self.dcov_func_sqexp(theta, x, mode= mode)
        elif covariance_function == 'mat52':
            self.cov_func = lambda theta, x, mode : self.cov_func_mat52(theta, x, mode=mode)
            self.dcov_func = lambda theta, x, mode : self.dcov_func_mat52(theta, x, mode= mode)
        elif covariance_function == 'mat72':
            self.cov_func = lambda theta, x, mode: self.cov_func_mat72(theta, x, mode=mode)
            self.dcov_func = lambda theta, x, mode: self.dcov_func_mat72(theta, x, mode=mode)

    def logp_and_gradlogp(self, theta, y):
        """
        Returns the negative log (marginal) likelihood (the function to be optimised) and its gradient
        """
        #tmp is Ky
        tmp = self.cov_func(theta, self.XX, mode="Noise")
        #print "Ky shape=", tmp.shape
        #tmp is L
        try:
            tmp = np.linalg.cholesky(tmp)
            #print "L shape=",tmp.shape
        except:
            logp = 1.0e8
            dlogp = np.ones(theta.size)*1.0e8
            return logp, dlogp
        detK = 2.0*np.sum(np.log(np.diag(tmp)))
        #tmp is Linv
        tmp = np.linalg.inv(tmp)
        #print "Linv shape=",tmp.shape
        #tmp2 is Linvy
        tmp2 = np.dot(tmp,y)
        #print "Linvy shape=",tmp2.shape
        logp = np.dot(tmp2.T,tmp2)/2.0 + detK/2.0 + self.N*np.log(2*np.pi)/2.0
        nhypers = theta.size
        dlogp = np.zeros(nhypers)
        #tmp is Kinv
        tmp = np.dot(tmp.T,tmp)
        #print "Kinv shape=",tmp.shape
        #tmp2 becomes Kinvy
        tmp2 = np.reshape(np.dot(tmp,y),(self.N,1))
        #print "Kinvyshape=",tmp.shape
        #tmp2 becomes aaT
        tmp2 = np.dot(tmp2,tmp2.T)
        #tmp2 becomes Kinv - aaT
        tmp2 = tmp - tmp2
        dKdtheta = self.dcov_func(theta,self.XX,mode=0)
        dlogp[0] = np.sum(self.diag_dot(tmp2,dKdtheta))/2.0
        dKdtheta = self.dcov_func(theta,self.XX,mode=1)
        dlogp[1] = np.sum(self.diag_dot(tmp2,dKdtheta))/2.0
        dKdtheta = self.dcov_func(theta,self.XX,mode=2)
        dlogp[2] = np.sum(self.diag_dot(tmp2,dKdtheta))/2.0
        return logp, dlogp

    def cov_func_mat52(self, theta, x, mode="Noise"):
        if mode != "Noise":
            return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2))
        else:
            return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (
            1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2)) + theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func_mat52(self, theta, x, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        """
        if mode == 0:
            return 2*self.cov_func_mat52(theta, x, mode='nn')/theta[0]
        elif mode == 1:
            return np.sqrt(5)*np.abs(x)*self.cov_func_mat52(theta, x, mode='nn')/theta[1]**2 + theta[0] ** 2 * \
                        np.exp(-np.sqrt(5) * np.abs(x) / theta[1])*(-np.sqrt(5) * np.abs(x) / theta[1]**2 - 10 * np.abs(x) ** 2 / (3 * theta[1] ** 3))
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def cov_func_mat72(self, theta, x, mode="Noise"):
        if mode != "Noise":
            return theta[0]**2 * np.exp(-np.sqrt(7) * np.abs(x) / theta[1]) * (1 + np.sqrt(7) * np.abs(x) / theta[1] +
                                14 * np.abs(x)**2/(5 * theta[1]**2) + 7*np.sqrt(7)*np.abs(x)**3/(15*theta[1]**3))
        else:
            return theta[0]**2 * np.exp(-np.sqrt(7) * np.abs(x) / theta[1]) * (1 + np.sqrt(7) * np.abs(x) / theta[1]
                            + 14 * np.abs(x) ** 2 / (5 * theta[1] ** 2) + 7*np.sqrt(7)*np.abs(x)**3/(15*theta[1]**3)) +\
                            theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func_mat72(self, theta, x, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        """
        if mode == 0:
            return 2*self.cov_func_mat72(theta, x, mode='nn')/theta[0]
        elif mode == 1:
            return np.sqrt(7)*np.abs(x)*self.cov_func_mat72(theta, x, mode='nn')/theta[1]**2 + theta[0] ** 2 * \
                        np.exp(-np.sqrt(7) * np.abs(x) / theta[1])*(-np.sqrt(7) * np.abs(x) / theta[1]**2 - 28 *
                        np.abs(x) ** 2 / theta[1] ** 3 - 21*np.sqrt(7)*np.abs(x)**3 / theta[1]**4)
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def cov_func_sqexp(self, theta, x, mode="Noise"):
        """
        Covariance function including noise variance
        """
        if mode != "Noise":
            #Squared exponential
            return theta[0]**2.0*np.exp(-x**2.0/(2.0*theta[1]**2.0))
        else:
            #Squared exponential
            return theta[0]**2*np.exp(-x**2.0/(2.0*theta[1]**2.0)) + theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func_sqexp(self, theta, x, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        """
        if mode == 0:
            return 2*theta[0]*np.exp(-x**2/(2*theta[1]**2))
        elif mode == 1:
            return x**2*theta[0]**2*np.exp(-x**2/(2*theta[1]**2))/theta[1]**3
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def meanf(self, theta, y):
        """
        Posterior mean function
        """
        Kp = self.cov_func(theta,self.XXp, mode="nn")
        Ky = self.cov_func(theta,self.XX, mode="Noise")
        L = np.linalg.cholesky(Ky)
        Linv = soltri(L.T,np.eye(y.size)).T
        LinvKp = np.dot(Linv,Kp)
        return np.dot(LinvKp.T,np.dot(Linv,y))

    def covf(self,theta,XX,XXp,XXpp):
        """
        Posterior covariance matrix
        """
        Kp = self.cov_func(theta,self.XXp,mode="nn")
        Kpp = self.cov_func(theta,self.XXpp,mode="nn")
        Ky = self.cov_func(theta,self.XX)
        L = np.linalg.cholesky(Ky)
        Linv = np.linalg.inv(L)
        LinvKp = np.dot(Linv,Kp)
        return Kpp - np.dot(LinvKp.T,LinvKp)

    def trainGP(self, theta0, y):
        # Set bounds for hypers (they must be strictly positive)
        bnds = ((1e-5, None), (1e-5, None), (1e-3, None))

        # Do optimisation
        thetap = opt.fmin_l_bfgs_b(self.logp_and_gradlogp, theta0, fprime=None, args=(y,), bounds=bnds) #, factr=1e10, pgtol=0.1)

        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return thetap[0]

    def EvalGP(self, theta0, y):
        theta = self.trainGP(theta0, y)

        return self.meanf(theta, y), theta