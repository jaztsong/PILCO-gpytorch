import numpy as np
import torch
import time

from .mgpr import MGPR
from .. import controllers
from .. import rewards
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.set_default_dtype(torch.float32)


class PILCO(torch.nn.Module):
    def __init__(self, X, Y, num_induced_points=None, horizon=30, controller=None,
                reward=None, m_init=None, S_init=None, name=None):
        super(PILCO, self).__init__()
        if not num_induced_points:
            self.mgpr = MGPR(X, Y)
        else:
            self.mgpr = SMGPR(X, Y, num_induced_points)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(self.state_dim, self.control_dim)
        else:
            self.controller = controller

        if reward is None:
            self.reward = rewards.ExponentialReward(self.state_dim)
        else:
            self.reward = reward

        if m_init is None or S_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then define it as the first state in the dataset.
            self.m_init = X[0:1, 0:self.state_dim]
            self.S_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.S_init = S_init
        self.optimizer = None
        self.best_reward = 0



    def optimize_models(self, maxiter=100, restarts=1):
        '''
        Optimize GP models
        '''
        self.mgpr.optimize(restarts=restarts, training_iter = maxiter)
        # Print the resulting model parameters
        # ToDo: only do this if verbosity is large enough
        lengthscales = self.mgpr.model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().squeeze()
        variance = self.mgpr.model.covar_module.outputscale.cpu().detach().numpy().squeeze()
        noise = self.mgpr.model.likelihood.noise.cpu().detach().numpy().squeeze()
        print('-----Learned models------')
        print('---Lengthscales---\n',lengthscales)
        print('---Variances---\n',variance)
        print('---Noises---\n',noise)

    def optimize_policy(self, maxiter=12, restarts=1):
        '''
        Optimize controller's parameter's
        '''

        if self.optimizer == None:
            self.controller.randomize()
            self.optimizer = torch.optim.Adam([
                {'params':self.controller.parameters()},
                ], lr=5e-1)

        start = time.time()
        m = torch.tensor(self.m_init).float().cuda()
        s = torch.tensor(self.S_init).float().cuda()
        current_reward = self.compute_reward()
        current_params = self.controller.state_dict()
        reward = torch.zeros(1).float().cuda()
        for i in range(maxiter):
            self.optimizer.zero_grad()
            reward = self.predict(m,s,self.horizon)[2]
            loss = -reward
            loss.backward()
            # plot_grad_flow_v2(self.controller.parameters())
            print('(Optimize Policy: Iter %d/%d - Loss: %.3f)' % (i,maxiter,loss.item()))
            self.optimizer.step()

        end = time.time()
        print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, reward))


    def compute_action(self, x_m):
        x_m = torch.tensor(x_m).float().cuda()
        x_s = torch.zeros((self.state_dim, self.state_dim)).float().cuda()
        return self.controller.compute_action(x_m, x_s )[0]

    def predict(self, m_x, s_x, n):
        m = m_x
        s = s_x
        self.mgpr.model.eval()
        self.mgpr.likelihood.eval()
        reward = torch.zeros(1).float().cuda()
        for _ in range(n):
            m, s = self.propagate(m,s)
            reward = reward + self.reward.compute_reward(m,s)[0]

        return m, s, reward

    def propagate(self, m_x, s_x):
        # m_x = torch.tensor(m_x)
        # s_x = torch.tensor(s_x)
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        # m_u = torch.tensor(m_u)
        # s_u = torch.tensor(s_u)
        # c_xu = torch.tensor(c_xu)

        m = torch.cat((m_x, m_u), 1)
        s1 = torch.cat((s_x, s_x@c_xu), 1)
        s2 = torch.cat(((s_x@c_xu).t(), s_u), 1)
        s = torch.cat((s1, s2), 0)

        # M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_dx, S_dx, C_dx = self.mgpr(m, s)

        # M_dx = torch.tensor(M_dx)
        # S_dx = torch.tensor(S_dx)
        # C_dx = torch.tensor(C_dx)

        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + C_dx.t() @ s1.t()

        # While-loop requires the shapes of the outputs to be fixed
        # M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        return self.predict(self.m_init,self.S_init,self.horizon)[2]
