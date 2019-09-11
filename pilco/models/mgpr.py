import math
import torch
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
import numpy as np
import os
import matplotlib.pyplot as plt
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
# torch.set_default_dtype(torch.float32)
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=train_y.shape[0])
        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[2],
                    # lengthscale_prior = gpytorch.priors.GammaPrior(1,10),
                    batch_size=train_y.shape[0]),
                batch_size=train_y.shape[0],
                # outputscale_prior = gpytorch.priors.GammaPrior(1.5,2),
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MGPR(torch.nn.Module):
    def __init__(self, X, Y, learning_rate=10e-4): # The input requires X as a nxd tensor and Y is 1xn tensor
        super(MGPR, self).__init__()
        self.num_outputs = Y.shape[1]
        self.num_dims = X.shape[1]
        self.num_datapoints = X.shape[0]
        self.lr = learning_rate
        self.Y = torch.from_numpy(Y).float()
        self.Y = self.Y.t()
        self.X = torch.from_numpy(X).float()
        self.X = self.X.repeat(self.Y.shape[0],1,1)
        self.create_model(self.X, self.Y)
        self.cuda = True
    


    def create_model(self, X, Y):
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()
        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                # noise_prior=gpytorch.priors.GammaPrior(2,1.5),
                batch_size=Y.shape[0])
        self.model = ExactGPModel(X, Y, self.likelihood)
        self.likelihood.cuda()
        self.model.cuda()

    def set_XY(self,X,Y):
        self.Y = torch.from_numpy(Y).float()
        self.Y = self.Y.t()
        self.X = torch.from_numpy(X).float()
        self.X = self.X.repeat(self.Y.shape[0],1,1)
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()
        self.model.set_train_data(self.X,self.Y,strict=False)

    def optimize(self,restarts=1, training_iter = 200):
        self.likelihood.train()
        self.model.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=self.lr)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.X)
             # Calc loss and backprop gradients
            loss = -mll(output, self.Y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()



    def _generate_sigma_points(self,n,x,P,alpha=0.3,beta=2.0,kappa=0.1):
        if n != x.size()[0]:
            raise ValueError("expected size(x) {}, but size is {}".format(
                n, x.size()[0]))


        lambda_ = alpha**2 * (n + kappa) - n
        U = torch.cholesky((lambda_ + n)*P)

        sigmas = torch.zeros((2*n+1, n)).float().cuda()
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = x + U[k]
            sigmas[n+k+1] = x - U[k]

        return sigmas

    # def predict_on_noisy_inputs(self, m, s, num_samps=50000):
    def forward(self, m, s):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance

        We adopt the sampling approach by leveraging the power of GPU
        """
        assert(m.shape[1] == self.num_dims and s.shape == (self.num_dims,self.num_dims))
        self.likelihood.eval()
        self.model.eval()

        if type(m) != torch.Tensor:
            m = torch.tensor(m).float().cuda()
            s = torch.tensor(s).float().cuda()
            print("Warning: gradient may break in mgpr.predict_on_noisy_inputs")

        inv_s = torch.inverse(s)

        # MCMC sampling approach
        # num_samps = 500
        # sample_model = torch.distributions.MultivariateNormal(m[0],s)
        # pred_inputs = sample_model.rsample((num_samps,)).float()
        # # pred_inputs[pred_inputs != pred_inputs] = 0
        # # pred_inputs,_ = torch.sort(pred_inputs,dim=0)
        # pred_inputs = pred_inputs.reshape(num_samps,self.num_dims).repeat(self.num_outputs,1,1)

        # Unscented Transform
        pred_inputs = self._generate_sigma_points(m.shape[1],m[0],s)
        pred_inputs = pred_inputs.repeat(self.num_outputs,1,1)
        

        #centralize X ?
        # self.model.set_train_data(self.centralized_input(m),self.Y)
        # self.model.set_train_data(self.X,self.Y)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_outputs = self.model(pred_inputs)


        # Plot graph to do a visual debug
        # for i in range(self.Y.shape[0]):
        #     lower, upper = pred_outputs.confidence_region()
        #     plt.plot(self.X[0,:,0].detach().cpu().numpy(),self.Y[i].detach().cpu().numpy(),'r*',label='train')
        #     plt.plot(pred_inputs[0,:,0].detach().cpu().numpy(),pred_outputs.mean[i].detach().cpu().numpy(),'ko',label='test')
        #     plt.fill_between(pred_inputs[0,:,0].detach().cpu().numpy(),lower[i].detach().cpu().numpy(),upper[i].detach().cpu().numpy())
        #     plt.legend()
        #     plt.show()

        # M = np.mean(pred_outputs.mean.cpu().numpy(),axis=1)[None,:]
        # S = np.cov(pred_outputs.mean.cpu().numpy())
        # covs = np.cov(pred_inputs[0].t().cpu().numpy(),pred_outputs.mean.cpu().numpy())
        # V = (covs[0:self.num_dims,self.num_dims:])
        # V = inv_s @ V

        outputs = pred_outputs.rsample(torch.Size([500])).mean(0)
        # outputs = pred_outputs.mean
        M = torch.mean(outputs,1)[None,:]
        V_ = torch.cat((pred_inputs[0].t(),outputs),0)
        fact = 1.0 / (V_.size(1) - 1)
        V_ -= torch.mean(V_, dim=1, keepdim=True)
        V_t = V_.t()  # if complex: mt = m.t().conj()
        covs =  fact * V_.matmul(V_t).squeeze()
        # covs = np.cov(pred_inputs[0].t().cpu().numpy(),pred_outputs.mean.cpu().numpy())
        V = covs[0:self.num_dims,self.num_dims:]
        V = inv_s @ V
        S = covs[self.num_dims:,self.num_dims:]

        self.likelihood.train()
        self.model.train()

        return M, S, V

    def predict_y(self,X):
        self.likelihood.eval()
        self.model.eval()
        X = torch.from_numpy(X).float()
        X = X.repeat(self.Y.shape[0],1,1)
        X = X.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Y = self.model(X)

        return Y

    def centralized_input(self, m):
        if self.cuda == True:
            m = torch.tensor(m).float().cuda()
        return self.X - m

