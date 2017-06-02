#!/usr/bin/env python
"""

First attempt at constructing a spatio temporal GP with 2D spatial sections and a time axis
Here we are assuming each spatial slice has the same coordinates which allows efficient evaluation using 
the kronecker product.

"""

import numpy as np
from scipy import optimize as opt

class STGP(object):
	def __init__(self, t, x, tp, xp):
		"""

		:param t: T x 1 vector of input times 
		:param x: N x 2 vector of input spatial locations
		"""
		self.t = t
		self.T = t.size  # number of time points
		self.x = x
		self.N = x.shape[1]  # number of spatial points
		self.M = self.T * self.N

		# get abs diff of spatial locs and evaluate cov func
		self.XX = self.abs_diff_x(x, x, grid=False)
		self.XXp = self.abs_diff_x(xp, x, mode=1, grid=False)
		# self.XXpp = self.abs_diff_x(xp, xp)

		# get abs diff of temporal locs and evaluate cov func
		self.TT = self.abs_diff_t(t, t)
		self.TTp = self.abs_diff_t(tp, t)

	# self.TTpp = self.abs_diff_t(tp, tp)

	def train_GP(self, theta0, y):
		# Set bounds for hypers (they must be strictly positive)
		bnds = ((1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None))

		# Do optimisation
		thetap = opt.fmin_l_bfgs_b(self.logp, theta0, fprime=None, args=(y,), bounds=bnds, pgtol=1e-8, factr=1e4)

		# Check for convergence
		if thetap[2]["warnflag"]:
			print "Warning flag raised"
		# Return optimised value of theta
		return thetap[0]

	def diag_dot(self, A, B):
		"""
		Computes the diagonal of C = AB where A and B are square matrices
		"""
		D = np.zeros(A.shape[0])
		for i in xrange(A.shape[0]):
			D[i] = np.dot(A[i, :], B[:, i])
		return D

	def logp(self, theta, y):
		Kx = self.cov_func(theta[0:3], self.XX)
		Kt = self.cov_func(theta[3::], self.TT)
		Lx = np.linalg.cholesky(Kx)
		Lxinv = np.linalg.inv(Lx)
		Kxinv = np.dot(Lxinv.T, Lxinv)
		Lt = np.linalg.cholesky(Kt)
		Ltinv = np.linalg.inv(Lt)
		Ktinv = np.dot(Ltinv.T, Ltinv)
		detKx = 2.0 * np.sum(np.log(np.diag(Lx)))
		detKt = 2.0 * np.sum(np.log(np.diag(Lt)))
		detK = self.T * detKx + self.N * detKt
		Linv = np.kron(Lxinv, Ltinv)
		Linvy = np.dot(Linv, y)
		logp = np.dot(Linvy.T, Linvy) / 2.0 + detK / 2.0 + self.M * np.log(2 * np.pi) / 2.0
		# get the derivative w.r.t. theta
		# tmp is Kinv
		Kinv = np.kron(Kxinv, Ktinv)
		# tmp2 becomes Kinvy
		Kinvy = np.reshape(np.dot(Kinv, y), (self.M, 1))
		# tmp2 becomes aaT
		aaT = np.dot(Kinvy, Kinvy.T)
		# tmp2 becomes Kinv - aaT
		tmp2 = Kinv - aaT
		nhypers = theta.size
		dlogp = np.zeros(nhypers)
		for i in xrange(nhypers):
			dKdtheta = self.dcov_func(theta, self.XX, self.TT, Kx, Kt, mode=i)
			dlogp[i] = np.sum(self.diag_dot(tmp2, dKdtheta)) / 2.0
		print logp, dlogp
		return logp, dlogp

	def meanf(self, theta, y):
		Kx = self.cov_func(theta[0:3], self.XX)
		Kt = self.cov_func(theta[3::], self.TT)
		Lx = np.linalg.cholesky(Kx)
		Lxinv = np.linalg.inv(Lx)
		Lt = np.linalg.cholesky(Kt)
		Ltinv = np.linalg.inv(Lt)
		Linv = np.kron(Lxinv, Ltinv)

		Kpx = self.cov_func(theta[0:3], self.XXp, mode=1)
		# Kppx = self.cov_func(theta[0:3], self.XXpp, mode=1)

		Kpt = self.cov_func(theta[3::], self.TTp, mode=1)
		# Kppt = self.cov_func(theta[3::], self.TTpp, mode=1)

		Kp = np.kron(Kpx, Kpt)
		# Kpp = np.kron(Kppx, Kppt)

		KpLinvT = np.dot(Kp, Linv.T)
		return np.dot(KpLinvT, np.dot(Linv, y))

	def abs_diff_t(self, x, xp):
		"""
		Creates matrix of differences (x_i - x_j) for vectorising.
		"""
		N = x.size
		Np = xp.size
		return np.tile(x, (Np, 1)).T - np.tile(xp, (N, 1))

	def abs_diff_x(self, x, xp, mode="same", grid=True):
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
			l, m = np.meshgrid(x[0, :], x[1, :])
			tmp = np.vstack((l.flatten(order='C'), m.flatten(order='C')))
			del l, m  # To save memory

			# Get number of data points
			N = x.shape[1] ** 2
		else:
			tmp = x
			N = x.shape[1]

		# abs differences
		XX = tmp[0, :] + 1j * tmp[1, :]  # complex trick to get vectorised differences in 2D
		# XX = x[0,:] + 1j*x[1,:]
		if mode == "same":
			XX = np.abs(np.tile(XX, (N, 1)).T - np.tile(XX, (N, 1)))
			return XX
		else:
			# Create grid and flatten
			# l,m = np.meshgrid(xp[0,:],xp[1,:])
			# tmp = np.vstack((l.flatten(order='C'),m.flatten(order='C')))
			# del l, m #To save memory
			Np = xp.shape[1]

			XXp = xp[0, :] + 1j * xp[1, :]  # complex trick to get vectorised differences in 2D
			XXp = np.abs(np.tile(XX, (Np, 1)).T - np.tile(XXp, (N, 1)))
			return XXp

	def cov_func(self, theta, x, mode="Noise"):
		"""
		Covariance function
		"""
		if mode != "Noise":
			# Squared exponential
			return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))
		else:
			# Squared exponential
			return theta[0] ** 2 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * np.eye(x.shape[0])

	def dcov_func(self, theta, x, t, Kx, Kt, mode=0):
		if mode == 0:
			return np.kron(2 * self.cov_func(theta[0:3], x, mode=1) / theta[0], Kt)
		elif mode == 1:
			return np.kron(x ** 2 * self.cov_func(theta[0:3], x, mode=1) / theta[1] ** 3, Kt)
		elif mode == 2:
			return np.kron(2 * theta[2] * np.eye(x.shape[0]), Kt)
		elif mode == 3:
			return np.kron(Kx, (2 * self.cov_func(theta[3::], t, mode=1) / theta[3]))
		elif mode == 4:
			return np.kron(Kx, (t ** 2 * self.cov_func(theta[3::], t, mode=1) / theta[4] ** 3))
		elif mode == 5:
			return np.kron(Kx, 2 * theta[5] * np.eye(t.shape[0]))