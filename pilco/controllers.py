import torch
import numpy as np
import gpytorch
from torch.autograd import Variable
from torch.nn import Parameter

from pilco.models import MGPR

def squash_sin(m, s, max_action=None):
    '''
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    '''
    k = m.shape[1]
    if s.shape == ():
        s = s.reshape(1,)

    if max_action is None:
        max_action = torch.ones((1,k)).float().cuda()  #squashes in [-1,1] by default
    else:
        max_action = max_action * torch.ones((1,k)).float().cuda()

    M = max_action * torch.exp(-torch.diag(s) / 2) * torch.sin(m)

    lq = -( torch.diag(s)[:, None] + torch.diag(s)[None, :]) / 2
    q = torch.exp(lq)
    S = (torch.exp(lq + s) - q) * torch.cos(torch.t(m) - m) \
        - (torch.exp(lq - s) - q) * torch.cos(torch.t(m) + m)
    S = max_action * torch.t(max_action) * S / 2

    tmp = torch.exp(-torch.diag(s)/2) * torch.cos(m)

    if tmp.dim() > 1:
        tmp = tmp.squeeze(0)
    C = max_action * torch.diag(tmp).squeeze()

    if S.dim() > 2:
        S = S.squeeze(-1)
    return M, S, C.reshape(k,k)


class LinearController(torch.nn.Module):
    def __init__(self, state_dim, control_dim, max_action=None):
        # gpflow.Parameterized.__init__(self)
        super(LinearController, self).__init__()
        self.W = Parameter(torch.rand(control_dim, state_dim).float().cuda(),requires_grad=True)
        self.b = Parameter(torch.rand(1, control_dim).float().cuda(),requires_grad=True)
        self.max_action = max_action

    def compute_action(self, m, s, squash=True):
        '''
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        if type(m) != torch.Tensor:
            m = torch.tensor(m).float().cuda()
            print("Warning: gradient may break in controller.compute_action")
        if type(s) != torch.Tensor:
            s = torch.tensor(s).float().cuda()
            print("Warning: gradient may break in controller.compute_action")

        M = m @ self.W.t() + self.b # mean output
        S = self.W @ s @ self.W.t() # output variance
        V = self.W.t() #input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self):
        mean = 0; sigma = 1
        with torch.no_grad():
            self.W.data.normal_(mean,sigma)
            self.b.data.normal_(mean,sigma)


class FakeGPR(torch.nn.Module):
    def __init__(self, X, Y, kernel):
        super(FakeGPR, self).__init__()
        self.X = Parameter(torch.tensor(X).float().cuda(),requires_grad=True)
        self.Y = Parameter(torch.tensor(Y).float().cuda(),requires_grad=True)
        self.covar_module = kernel
        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                # noise_prior=gpytorch.priors.GammaPrior(2,1.5),
                batch_shape=torch.Size([Y.shape[0]]))

    def set_train_data(self, X, Y, strict=None):
        self.X = Parameter(torch.tensor(X).float().cuda(),requires_grad=True)
        self.Y = Parameter(torch.tensor(Y).float().cuda(),requires_grad=True)


class RbfController(MGPR):
    '''
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    '''
    def __init__(self, state_dim, control_dim, num_basis_functions, max_action=None):
        MGPR.__init__(self,
            np.random.randn(num_basis_functions, state_dim),
            0.1*np.random.randn(num_basis_functions, control_dim),
            standarilze=False
        )

        # Remove the scale kernel which is the variance
        self.max_action = max_action

    def create_model(self, X, Y):
        kern = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.num_dims,
                    # lengthscale_prior = gpytorch.priors.GammaPrior(1,10),
                    batch_shape=torch.Size([self.num_outputs])),
                batch_shape=torch.Size([self.num_outputs]),
                outputscale_constraint=gpytorch.constraints.Interval(1.0,1.0+1e-5),
                # outputscale_prior = gpytorch.priors.GammaPrior(1.5,2),
                )
        self.model = FakeGPR(X, Y, kern)
        self.model.cuda()

    def optimize(self,restarts=1, training_iter = 200):
        raise NotImplementedError

    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        iK, beta = self.calculate_factorizations()
        M, S, V = self.predict_given_factorizations(m, s, 0.0 * iK, beta)
        S = S - torch.diag(self.model.covar_module.outputscale - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2

        return M, S, V

    def randomize(self):
        print("Randomising controller")
        with torch.no_grad():
            self.X.data.normal_(0,1)
            self.Y.data.normal_(0,0.1) 
            self.model.covar_module.base_kernel.lengthscale.normal_(0,1)

