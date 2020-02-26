import numpy as np
import sys
sys.path.append("/home/song3/Research/PILCO-gpytorch")
import pilco

def rollout(env, pilco, timesteps, verbose=False, random=False, SUBS=1, render=True):
    X = []; Y = []
    x = env.reset()
    print("Starting State:\n", x)
    for timestep in range(timesteps):
        if render: env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done: break
            if render: env.render()
        if verbose:
            print("Action: ", u)
            print("State : ", x_new)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
        if done: 
            print("Env Done...",timestep)
            break
    return np.stack(X), np.stack(Y)


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        u = pilco.compute_action(x[None, :])[0, :]
        return u.detach().cpu().numpy()



