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
    if tmp.shape == ():
        tmp = tmp.reshape(1,)
    C = max_action * torch.diag(tmp).squeeze()

    if S.dim() > 2:
        S = S.squeeze(-1)
    return M, S, C.reshape(k,k)


class LinearController(torch.nn.Module):
    def __init__(self, state_dim, control_dim, max_action=None):
        # gpflow.Parameterized.__init__(self)
        super(LinearController, self).__init__()
        self.W = Variable(torch.rand(control_dim, state_dim).float().cuda(),requires_grad=True)
        self.b = Variable(torch.rand(1, control_dim).float().cuda(),requires_grad=True)
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
        self.W.data = mean + sigma*torch.randn(self.W.shape)
        self.b = mean + sigma*torch.randn(self.b.shape)


# class FakeGPR(gpflow.Parameterized):
#     def __init__(self, X, Y, kernel):
#         gpflow.Parameterized.__init__(self)
#         self.X = gpflow.Param(X)
#         self.Y = gpflow.Param(Y)
#         self.kern = kernel
#         self.likelihood = gpflow.likelihoods.Gaussian()

class RbfController(MGPR):
    '''
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    '''
    def __init__(self, state_dim, control_dim, num_basis_functions, max_action=None):
        MGPR.__init__(self,
            np.random.randn(num_basis_functions, state_dim),
            0.1*np.random.randn(num_basis_functions, control_dim)
        )

        # Remove the scale kernel which is the variance
        self.model.covar_module = self.model.covar_module.base_kernel
        self.max_action = max_action


    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        M, S, V = self.predict_on_noisy_inputs(m, s)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2

        return M, S, V

    def randomize(self):
        print("Randomising controller")
        self.model.covar_module.lengthscale = torch.rand(self.num_outputs,1,self.num_dims).cuda()

        X = torch.randn(self.num_datapoints,self.num_dims).repeat(self.num_outputs,1,1).cuda()
        Y = 0.1*torch.randn(self.num_outputs,self.num_datapoints).cuda()
        self.model.set_train_data(X,Y)

