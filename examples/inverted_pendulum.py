import numpy as np
import sys
sys.path.append("/home/song3/Research/PILCO-gpytorch")
import roboschool,gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import matplotlib.pyplot as plt
np.random.seed(0)
import torch

from utils import rollout, policy

env = gym.make('RoboschoolInvertedPendulum-v1')
# env = gym.make('CartPole-v0')
# Initial random rollouts to generate a dataset
X,Y = rollout(env=env, pilco=None, random=True, timesteps=40)
for i in range(1,3):
    X_, Y_ = rollout(env=env, pilco=None, random=True,  timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
# controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
controller = LinearController(state_dim=state_dim, control_dim=control_dim)

# pilco = PILCO(X, Y, controller=controller, horizon=40)
# Example of user provided reward function, setting a custom target state
R = ExponentialReward(state_dim=state_dim, t=np.array([0.0,0.0,1.0,0.1,0.0]))
pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

# Example of fixing a parameter, optional, for a linear controller only
#pilco.controller.b = np.array([[0.0]])
#pilco.controller.b.trainable = False
m_init = np.reshape([0.0, 0.1,1.0,0.038, -0.2], (1,5))
S_init = np.diag([0.01, 0.01, 0.01,0.01,0.01])
m_init = torch.from_numpy(m_init).float().cuda()
S_init = torch.from_numpy(S_init).float().cuda()
T = 30 

for rollouts in range(3):
    pilco.optimize_models()
    pilco.optimize_policy()

    X_new, Y_new = rollout(env=env, pilco=pilco, timesteps=100,render=True)

    # multi-step prediction
    m_p = np.zeros((T, state_dim))
    S_p = np.zeros((T, state_dim, state_dim))
    for h in range(T):
        m_h, S_h, _ = pilco.predict(m_init, S_init,h)
        m_p[h,:], S_p[h,:,:] = m_h[0,:].detach().cpu().numpy(), S_h[:,:].detach().cpu().numpy()
	

    for i in range(state_dim):    
        plt.plot(range(T-1), m_p[0:T-1, i], X_new[1:T, i]) # can't use Y_new because it stores differences (Dx)
        plt.fill_between(range(T-1),
                m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
                m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.2)
        plt.show()


#     pred_outputs = pilco.mgpr.predict_y(X_new)
#     for i in range(Y.shape[1]):
#         lower, upper = pred_outputs.confidence_region()
#         plt.plot(range(len(Y_new[:,i])), pred_outputs.mean[i].detach().cpu().numpy(),Y_new[:,i])
#         # plt.plot(range(len(Y[:,i])), Y[:,i],'ko')
#         plt.fill_between(range(len(Y_new[:,i])),lower[i].detach().cpu().numpy(),upper[i].detach().cpu().numpy(),alpha=0.3)
#         plt.show()

    print("One iteration done")
    import pdb;pdb.set_trace()
    # print("No of ops:", len(tf.get_default_graph().get_operations()))
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
