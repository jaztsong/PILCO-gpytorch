#! /home/song3/anaconda3/envs/pilco/bin/python 
import sys
sys.path.append("/home/song3/Research/PILCO-gpytorch")
from pilco.models import MGPR
import numpy as np
import os
import oct2py
octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
octave.addpath(dir_path)

def predict_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)

def test_predictions():
    np.random.seed(1)
    d = 3  # Input dimension
    k = 2  # Number of outputs
    n = 10 # number of datapoints

    # Training Dataset
    X0 = np.random.rand(n, d)
    A = np.random.rand(d, k)
    Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(n, k) - 0.5)  #  Just something smooth
    mgpr = MGPR(X0, Y0)

    mgpr.optimize()

    # Generate input
    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = mgpr.predict_on_noisy_inputs(m, s)

    # Change the dataset and predict again. Just to make sure that we don't cache something we shouldn't.
    # X0 = 5*np.random.rand(n, d)
    # mgpr.set_XY(X0, Y0) 

    # M, S, V = mgpr.predict_on_noisy_inputs(m, s)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = mgpr.model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().squeeze()
    variance = mgpr.model.covar_module.outputscale.cpu().detach().numpy().squeeze()
    noise = mgpr.model.likelihood.noise.cpu().detach().numpy().squeeze()

    hyp = np.log(np.hstack(
        (lengthscales,
         np.sqrt(variance[:, None]),
         np.sqrt(noise[:, None]))
    )).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0

    # Call function in octave
    M_mat, S_mat, V_mat = octave.gp0(gpmodel, m.T, s, nout=3)
    print(M - M_mat.T)
    print(S - S_mat)
    print(V - V_mat)
    import pdb;pdb.set_trace()
    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-4)
    np.testing.assert_allclose(S, S_mat, rtol=1e-4)
    np.testing.assert_allclose(V, V_mat, rtol=1e-4)

    print("lengthscal:\n",lengthscales)
    print("variance:\n",variance)
    print("noise:\n",noise)

if __name__ == '__main__':
    test_predictions()
