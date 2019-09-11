import sys
sys.path.append("/home/song3/Research/PILCO-gpytorch")
from pilco.rewards import ExponentialReward
import numpy as np
import os
import torch
import oct2py
octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
octave.addpath(dir_path)



def test_reward():
    '''
    Test reward function by comparing to reward.m
    '''
    k = 2  # state dim
    m = np.random.rand(1, k)
    s = np.random.randn(k, k)
    s = s.dot(s.T)

    reward = ExponentialReward(k)
    W = reward.W.data.cpu().numpy()
    t = reward.t.data.cpu().numpy()

    M, S = reward.compute_reward(torch.tensor(m).float().cuda(),
            torch.tensor(s).float().cuda())

    M_mat, _, _, S_mat = octave.reward(m.T, s, t.T, W, nout=4)

    np.testing.assert_allclose(M.cpu().numpy(), M_mat)
    np.testing.assert_allclose(S.cpu().numpy(), S_mat)


if __name__ == '__main__':
    test_reward()
