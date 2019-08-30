import math
import torch
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
import numpy as np
import os
import matplotlib.pyplot as plt
# torch.set_default_dtype(torch.float32)
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=train_y.shape[0])
        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[2],
                    batch_size=train_y.shape[0]),
                batch_size=train_y.shape[0]
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MGPR(torch.nn.Module):
    def __init__(self, X, Y, learning_rate=0.1): # The input requires X as a nxd tensor and Y is 1xn tensor
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
        self.model.set_train_data(self.X,self.Y)

    def optimize(self,restarts=1, training_iter = 50):
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

    def predict_on_noisy_inputs(self, m, s, num_samps=500):
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

        if self.cuda == True:
            m = torch.tensor(m).float().cuda()
            s = torch.tensor(s).float().cuda()
            inv_s = torch.inverse(s)

        sample_model = torch.distributions.MultivariateNormal(m,s)
        pred_inputs = sample_model.sample((num_samps,)).float()
        pred_inputs,_ = torch.sort(pred_inputs,dim=0)
        pred_inputs = pred_inputs.reshape(num_samps,self.num_dims).repeat(self.num_outputs,1,1)

        #centralize X ?
        # self.model.set_train_data(self.centralized_input(m),self.Y)
        # self.model.set_train_data(self.X,self.Y)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_outputs = (self.model(pred_inputs))


        # Plot graph to do a visual debug
        # lower, upper = pred_outputs.confidence_region()
        # plt.plot(self.X[0,:,0].cpu().numpy(),self.Y[0].cpu().numpy(),'k*',label='train')
        # plt.plot(pred_inputs[0,:,0].cpu().numpy(),pred_outputs.mean[0].cpu().numpy(),'ko',label='test')
        # plt.fill_between(pred_inputs[0,:,0].cpu().numpy(),lower[0].cpu().numpy(),upper[0].cpu().numpy())
        # plt.legend()
        # plt.show()

        # M = np.mean(pred_outputs.mean.cpu().numpy(),axis=1)[None,:]
        # S = np.cov(pred_outputs.mean.cpu().numpy())
        # covs = np.cov(pred_inputs[0].t().cpu().numpy(),pred_outputs.mean.cpu().numpy())
        # V = (covs[0:self.num_dims,self.num_dims:])
        # V = inv_s @ V

        M = torch.mean(pred_outputs.mean,1)[None,:]
        S = pred_outputs.mean.var(dim=1)
        V_ = torch.cat((pred_inputs[0].t(),pred_outputs.mean),0)
        fact = 1.0 / (V_.size(1) - 1)
        V_ -= torch.mean(V_, dim=1, keepdim=True)
        V_t = V_.t()  # if complex: mt = m.t().conj()
        covs =  fact * V_.matmul(V_t).squeeze()
        # covs = np.cov(pred_inputs[0].t().cpu().numpy(),pred_outputs.mean.cpu().numpy())
        V = (covs[0:self.num_dims,self.num_dims:])
        V = inv_s @ V

        return M, S, V


    def centralized_input(self, m):
        if self.cuda == True:
            m = torch.tensor(m).float().cuda()
        return self.X - m

