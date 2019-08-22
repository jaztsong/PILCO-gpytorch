import math
import torch
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
import numpy as np
import os
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
	super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
	self.mean_module = gpytorch.means.ConstantMean(batch_size=train_y[0].shape[1])
	self.covar_module = gpytorch.kernels.ScaleKernel(
	    gpytorch.kernels.RBFKernel(batch_size=train_y[0].shape[1],
			lengthscale_prior=gpytorch.priors.GammaPrior(1,10)),
		batch_size=train_y[0].shape[1]
	)

    def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MPGR(torch.nn.Module):
    def __init__(self, X, Y, learning_rate=0.1):
        super(MPGR, self).__init__(X, Y)
		self.num_outputs = X.shape[1]
        self.num_dims = X.shape[1]
        self.num_datapoints = X.shape[0]
		self.lr = learning_rate
		self.X = X
		self.Y = Y
        self.create_model(X, Y)
    


    def create_models(self, X, Y):
		# initialize likelihood and model
		self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
			noise_prior=gpytorch.priors.GammaPrior(1.5,2),
			batch_size=Y.shape[1])
		self.model = ExactGPModel(X, Y, likelihood)

	def set_XY(self,X,Y):
		self.model.set_train_data(X,Y)
		self.X = X
		self.Y = Y

	def optimize(self,training_iter = 50):
		self.likelihood.train()
		self.model.train()

		# Use the adam optimizer
		optimizer = torch.optim.Adam([
    		{'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
		], lr=self.lr)
		# "Loss" for GPs - the marginal log likelihood
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
		training_iter = 50
		for i in range(training_iter):
    		# Zero gradients from previous iteration
    		optimizer.zero_grad()
    		# Output from model
    		output = self.model(self.model.train_inputs)
   			 # Calc loss and backprop gradients
    		loss = -mll(output, self.model.train_targets).sum()
    		loss.backward()
    		print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    		optimizer.step()

	def predict_on_noisy_inputs(self, m, s):
		self.likelihood.eval()
		self.model.eval()
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

	def calculate_factorizations(self):
		'''
		K = self.K(self.X)
        batched_eye = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=float_type)
        L = tf.cholesky(K + self.noise[:, None, None]*batched_eye)
        iK = tf.cholesky_solve(L, batched_eye)
        Y_ = tf.transpose(self.Y)[:, :, None]
        # Why do we transpose Y? Maybe we need to change the definition of self.Y() or beta?
        beta = tf.cholesky_solve(L, Y_)[:, :, 0]
        return iK, beta
		'''

		K = self.K(self.model.train_inputs[0])
		batched_eye = torch.eye(model.train_inputs[0].shape[1]).repeat(model.train_targets.shape[0],1,1)
		L = psd_safe_cholesky(K + self.noise[:,None]*batched_eye)
		iK = torch.cholesky_solve(batched_eye, L)
		Y_ = model.train_targets.t()
		beta = torch.cholesky_solve(Y_, L)

		return iK, beta

	def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.matrix_diag(1/self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_dims, dtype=float_type)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.transpose(
                tf.matrix_solve(B, tf.linalg.transpose(iN), adjoint=True),
            )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.matrix_diag(
                1/tf.square(self.lengthscales[None, :, :]) +
                1/tf.square(self.lengthscales[:, None, :])
            ) + tf.eye(self.num_dims, dtype=float_type)

        # TODO: change this block according to the PR of tensorflow. Maybe move it into a function?
        X = inp[None, :, :, :]/tf.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :]/tf.square(self.lengthscales[None, :, None, :])
        Q = tf.matrix_solve(R, s)/2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + \
            Xs[:, :, :, None] + X2s[:, :, None, :]
        #
        k = tf.log(self.variance)[:, None] - \
            tf.reduce_sum(tf.square(iN), -1)/2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
                @ L @
                tf.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
            )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)


	def centralized_input(self, m):
        return self.X - m

	def K(self,X1,X2=None):
		return self.model.covar_module(X1,X2)

	def noise(self):
		return self.model.likelihood.noise
	
def test_predictions():
    np.random.seed(0)
    d = 3  # Input dimension
    k = 2  # Number of outputs

    # Training Dataset
    X0 = np.random.rand(100, d)
    A = np.random.rand(d, k)
    Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(100, k) - 0.5)  #  Just something smooth
    mgpr = MGPR(X0, Y0)

    mgpr.optimize()

    # Generate input
    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = mgpr.predict_on_noisy_inputs(m, s)

    # Change the dataset and predict again. Just to make sure that we don't cache something we shouldn't.
    X0 = 5*np.random.rand(100, d)
    mgpr.set_XY(X0, Y0) 

    M, S, V = mgpr.predict_on_noisy_inputs(m, s)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = np.stack([model.kern.lengthscales.value for model in mgpr.models])
    variance = np.stack([model.kern.variance.value for model in mgpr.models])
    noise = np.stack([model.likelihood.variance.value for model in mgpr.models])

if __name__ == '__main__':
    test_predictions()