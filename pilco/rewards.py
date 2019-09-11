import abc
import torch
import numpy as np
from torch import tensor
from torch.nn import Module,Parameter
float_type = torch.float32




class ExponentialReward(Module):
    def __init__(self, state_dim, W=None, t=None):
        Module.__init__(self)
        self.state_dim = state_dim
        if W is not None:
            self.W = Parameter(tensor(np.reshape(W, (state_dim, state_dim))), requires_grad=False).float().cuda()
        else:
            self.W = Parameter(tensor(np.eye(state_dim)), requires_grad=False).float().cuda()
        if t is not None:
            self.t = Parameter(tensor(np.reshape(t, (1, state_dim))), requires_grad=False).float().cuda()
        else:
            self.t = Parameter(tensor(np.zeros((1, state_dim))), requires_grad=False).float().cuda()

    def compute_reward(self, m, s):
        '''
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input m : [1, k]
        Input s : [k, k]

        Output M : [1, 1]
        Output S  : [1, 1]
        '''

        SW = s @ self.W

        X, LU = torch.solve(torch.t(self.W),(torch.eye(self.state_dim, dtype=float_type).cuda() + SW) )

        muR = torch.exp(-(m-self.t) @ torch.t(X) @ torch.t(m-self.t)/2) / \
                torch.sqrt(torch.det(torch.eye(self.state_dim, dtype=float_type).cuda() + SW))

        X, LU = torch.solve(torch.t(self.W),(torch.eye(self.state_dim, dtype=float_type).cuda() + 2 * SW) )

        r2 =  torch.exp(-(m-self.t) @ torch.t(X) @ torch.t(m-self.t)) / \
                torch.sqrt(torch.det(torch.eye(self.state_dim, dtype=float_type).cuda() + 2 * SW))

        sR = r2 - muR @ muR
        # muR.set_shape([1, 1])
        # sR.set_shape([1, 1])
        return muR.reshape((1,1)), sR.reshape((1,1))


class LinearReward(Module):
    def __init__(self, state_dim, W):
        Module.__init__(self)
        self.state_dim = state_dim
        self.W = Parameter(tensor(np.reshape(W, (state_dim, 1))), requires_grad=False)

    def compute_reward(self, m, s):
        muR = m @ self.W
        sR = torch.t(self.W) @ s @ self.W
        return muR, sR


class CombinedRewards(Module):
    def __init__(self, state_dim, rewards=[], coefs=None):
        Module.__init__(self)
        self.state_dim = state_dim
        self.base_rewards = rewards
        if coefs is not None:
            self.coefs = coefs
        else:
            self.coefs = np.ones(len(list))

    def compute_reward(self, m, s):
        muR = 0
        sR = 0
        for c,r in enumerate(self.base_rewards):
            tmp1, tmp2 = r.compute_reward(m, s)
            muR += self.coefs[c] * tmp1
            sR += self.coefs[c] ** 2 * tmp2
        return muR, sR
