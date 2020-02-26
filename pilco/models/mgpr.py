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
        self.num_out = train_y.shape[0]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_out]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[2],
                    # lengthscale_prior = gpytorch.priors.GammaPrior(1,10),
                    batch_shape=torch.Size([self.num_out])),
                batch_shape=torch.Size([self.num_out]),
                outputscale_constraint = gpytorch.constraints.Interval(0.001,0.001001),
                # outputscale_prior = gpytorch.priors.GammaPrior(1.5,2),
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        # return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
        #         gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        #         )


class MGPR(torch.nn.Module):
    def __init__(self, X, Y, learning_rate=8e-2, standarilze=True): # The input requires X as a nxd tensor and Y is 1xn tensor
        super(MGPR, self).__init__()
        self.num_outputs = Y.shape[1]
        self.num_dims = X.shape[1]
        self.num_datapoints = X.shape[0]
        self.lr = learning_rate
        self.standarilze = standarilze
        # self.Y = torch.from_numpy(Y).float()
        # self.Y = self.Y.t()
        # self.X = torch.from_numpy(X).float()
        # self.X = self.X.repeat(self.Y.shape[0],1,1)
        self.model = None
        self.set_XY(X,Y)
        self.create_model(self.X, self.Y)
        self.cuda = True
    


    def create_model(self, X, Y):
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()
        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                # noise_prior=gpytorch.priors.GammaPrior(2,1.5),
                batch_shape=torch.Size([Y.shape[0]]))
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
        # normailize the data: input and output
        if self.standarilze:
            self.X_mean = self.X.mean(dim=-2, keepdim=True)
            self.X_std = self.X.std(dim=-2, keepdim=True)
            self.Y_mean = self.Y.mean(dim=-1, keepdim=True)
            self.Y_std = self.Y.std(dim=-1, keepdim=True)
            self.X = (self.X - self.X_mean)/self.X_std
            self.Y = (self.Y - self.Y_mean)/self.Y_std
        if self.model != None:
            self.model.set_train_data(self.X,self.Y,strict=False)

    def optimize(self,restarts=1, training_iter = 200):
        self.likelihood.train()
        self.model.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)


        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=self.lr)
        # "Loss" for GPs - the marginal log likelihood
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



    def _generate_sigma_points(self,n,x,P,alpha=0.1,beta=2.0,kappa=3):
        if n != x.size()[0]:
            raise ValueError("expected size(x) {}, but size is {}".format(
                n, x.size()[0]))


        lambda_ = alpha**2 * (n + kappa) - n
        try:
            U = torch.cholesky((lambda_ + n)*P)
        except:
            U = torch.zeros((n,n)).float().cuda()



        sigmas = torch.zeros((2*n+1, n)).float().cuda()
        W_c = torch.ones(2*n+1).float().cuda()
        W_m = torch.ones(2*n+1).float().cuda()
        sigmas[0] = x
        W_c[0] = lambda_/(n+lambda_) + (1 - alpha**2 + beta)
        W_m[0] = lambda_/(n+lambda_)
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = x + U[k]
            sigmas[n+k+1] = x - U[k]
            W_c[2*k+1] = W_m[2*k+1] = 0.5/(lambda_ + n)
            W_c[2*k+2] = W_m[2*k+2] = 0.5/(lambda_ + n)


        return sigmas, W_m, W_c

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

        K = self.K(self.X)
        batched_eye = torch.eye(self.X.shape[1]).repeat(self.Y.shape[0],1,1).float().cuda()
        # L = psd_safe_cholesky(K + self.model.likelihood.noise[:,None]*batched_eye)
        # iK = torch.cholesky_solve(batched_eye, L)
        #work-around solution without cholesky_solve
        iK, _ = torch.solve(batched_eye, K + self.model.likelihood.noise[:,None]*batched_eye)
        Y_ = self.Y[:,:,None]
        # beta = torch.cholesky_solve(Y_, L)[:,:,0]
        #work-around solution without cholesky_solve
        beta, _ = torch.solve(Y_, K + self.model.likelihood.noise[:,None]*batched_eye)
        beta = beta[:,:,0]

        return iK, beta
    def predict_given_factorizations(self, m, s ,iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        if type(m) != torch.Tensor or type(s) != torch.Tensor:
            m = torch.tensor(m).float().cuda()
            s = torch.tensor(s).float().cuda()
            print("Warning: gradient may break in mgpr.predict_given_factorizations")

        s = s.repeat(self.num_outputs, self.num_outputs, 1, 1)
        inp = self.centralized_input(m)
        

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = torch.diag_embed(1/(self.model.covar_module.base_kernel.lengthscale.squeeze(1)))
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + torch.eye(self.num_dims).float().cuda()

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t,_ = torch.solve(torch.transpose(iN,dim0=1,dim1=2), B)
        t = torch.transpose(t, dim0=1,dim1=2)

        lb = torch.exp(-torch.sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        t_det = torch.det(B)

        c = self.model.covar_module.outputscale / torch.sqrt(t_det)

        M = (torch.sum(lb, -1) * c)[:, None]
        V = (torch.transpose(tiL, dim0=1,dim1=2) @ lb[:, :, None])[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R_0 = torch.diag_embed(
                1/torch.pow(self.model.covar_module.base_kernel.lengthscale.squeeze(1)[None,:,:],2) + 
                1/torch.pow(self.model.covar_module.base_kernel.lengthscale.squeeze(1)[:,None,:],2) 
                )
        R = s @ R_0 + torch.eye(self.num_dims).float().cuda()

        # TODO: change this block according to the PR of tensorflow. Maybe move it into a function?
        X = inp[None, :, :, :]/torch.pow(self.model.covar_module.base_kernel.lengthscale.squeeze(1)[:, None, None, :],2)
        X2 = -inp[:, None, :, :]/torch.pow(self.model.covar_module.base_kernel.lengthscale.squeeze(1)[None, :, None, :],2)
        q_x, _ = torch.solve(s, R)
        Q = q_x/2
        Xs = torch.sum(X @ Q * X, -1)
        X2s = torch.sum(X2 @ Q * X2, -1)
        maha = -2 * ((X @ Q) @ torch.transpose(X2,dim0=2,dim1=3)) + \
           Xs[:, :, :, None] + X2s[:, :, None, :]

        #
        k = torch.log(self.model.covar_module.outputscale)[:, None] - \
            torch.sum(torch.pow(iN,2), -1)/2
        L = torch.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = beta[:, None, None, :].repeat(1, self.num_outputs, 1, 1)
        S = (beta[:, None, None, :].repeat(1, self.num_outputs, 1, 1)
                @ L @
                beta[None, :, :, None].repeat(self.num_outputs, 1, 1, 1)
            )[:, :, 0, 0]

        diagL = torch.diagonal(L.permute((3,2,1,0)), dim1=-2,dim2=-1).permute(2,1,0)
        S = S - torch.diag_embed(torch.sum((iK * diagL),[1,2]))
        r_det = torch.det(R)

        S = S / torch.sqrt(r_det)
        S = S + torch.diag_embed(self.model.covar_module.outputscale)
        S = S - M @ M.t()

        return M.t(), S, V.t()
    
    # def predict_on_noisy_inputs(self, m, s, num_samps=50000):
    def forward(self, m, s, method='moment_matching'):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance

        We adopt the sampling approach by leveraging the power of GPU
        """
        assert(m.shape[1] == self.num_dims and s.shape == (self.num_dims,self.num_dims))
        #standarize
        if type(m) != torch.Tensor or type(s) != torch.Tensor:
            m = torch.tensor(m).float().cuda()
            s = torch.tensor(s).float().cuda()
            print("Warning: gradient may break in mgpr.predict_on_noisy_inputs")

        if self.standarilze:
            inv_s = torch.inverse(s)
            m = (m - self.X_mean[0])/self.X_std[0]
            s = s/(self.X_std[0].t()*self.X_std[0])

        if method == 'moment_matching':
            iK, beta = self.calculate_factorizations()
            M, S, V =  self.predict_given_factorizations(m,s,iK,beta)
            if self.standarilze:
                M = M*self.Y_std.t() + self.Y_mean.t()
                S = S*(self.Y_std*self.Y_std.t())
                var = torch.cat((self.X_std[0],self.Y_std.t()),dim=-1)
                mask_var = var.t()*var
                cov = s@V
                cov = cov*mask_var[0:self.num_dims,self.num_dims:]
                V = inv_s@cov
            return M, S, V

        ###########################################################################
        ###The following is the method of MCMC and Unscented Transformed###########

        # self.likelihood.eval()
        # self.model.eval()

        inv_s = torch.inverse(s)

        if method == 'unscented_transform':
            # Unscented Transform
            pred_inputs, W_m, W_c = self._generate_sigma_points(m.shape[1],m[0],s)
            pred_inputs = pred_inputs.repeat(self.num_outputs,1,1)
        else:
            # MCMC sampling approach
            num_samps = 200
            try:
                sample_model = gpytorch.distributions.MultivariateNormal(m[0],s)
            except:
                import pdb;pdb.set_trace()
                


            pred_inputs = sample_model.rsample(torch.Size([num_samps])).float()
            # pred_inputs[pred_inputs != pred_inputs] = 0
            # pred_inputs,_ = torch.sort(pred_inputs,dim=0)
            pred_inputs = pred_inputs.reshape(num_samps,self.num_dims).repeat(self.num_outputs,1,1)

        

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

        outputs = pred_outputs.rsample(torch.Size([10])).mean(0)
        # outputs = pred_outputs.mean
        if method == 'unscented_transform':
            M = torch.mean(W_m*outputs,1)[None,:]
        else:
            M = torch.mean(outputs,1)[None,:]
        V_ = torch.cat((pred_inputs[0].t(),outputs),0)
        fact = 1.0 / (V_.size(1) - 1)
        V_ -= torch.mean(V_, dim=1, keepdim=True)
        V_t = V_.t()  # if complex: mt = m.t().conj()
        if method == 'unscented_transform':
            V_ = V_*W_c
        covs =  fact * V_.matmul(V_t).squeeze()
        # covs = np.cov(pred_inputs[0].t().cpu().numpy(),pred_outputs.mean.cpu().numpy())
        V = covs[0:self.num_dims,self.num_dims:]
        V = inv_s @ V
        S = covs[self.num_dims:,self.num_dims:]


        return M, S, V

    def predict_y(self,X):
        # self.likelihood.eval()
        # self.model.eval()
        X = torch.from_numpy(X).float().cuda()
        X = X.repeat(self.Y.shape[0],1,1)
        if self.standarilze:
            X = (X - self.X_mean)/self.X_std

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Y = self.model(X)

        if self.standarilze:
            Y.loc = Y.loc*self.Y_std + self.Y_mean
            Y.covariance_matrix = self.Y_std[:,None]*self.Y_std[:,None]*Y.covariance_matrix
        return Y

    def centralized_input(self, m):
        if self.cuda == True:
            m = torch.tensor(m).float().cuda()
        return self.X - m

    def K(self,X1,X2=None):
        return self.model.covar_module(X1,X2).evaluate()

