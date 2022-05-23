from solver import Logistic_regression_L1_regularization, build_logistic_regression_ell1_regularization_problem_instance
import numpy as np
from matplotlib import pyplot as plt
import pickle
from time import time


# Define some parameters.
d = 100
n = 1000000
rho = 0.98
sketch_size1 = int(0.1 * n)             
trials = 3
alpha_backtrack, beta_backtrack = 0.1, 0.5 
A, y, B, b  = build_logistic_regression_ell1_regularization_problem_instance(n, d, rho)
regularization_parameter = 0.0001
max_iter = 200
mu = 10

_lambda0 = np.random.rand(5*d, 1)                       # Multiplier for non-negativity constraints.
nu0 = np.zeros((2*d, 1))                                # Multiplier for equality constraints.
w0 = np.random.rand(5*d, 1)                             # Initial guess. 


# Solve problem without sketching.
no_sketch_solver = Logistic_regression_L1_regularization(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, 0, 1)
tic1 = time()
w3, _lambda3, nu3, residual_norms_instance3, times3, alphas3, obj3, rd3, rcent3, rfeas3, residual_opt_conds3 = no_sketch_solver.solve(max_iter, w0, _lambda0, nu0, mu)
toc1 = time() - tic1
print("Time 1:", toc1)
_x = w3[0:d] - w3[d:2*d]
residual_opt_conds_sketching_instance2 = []
times_instance2 = []
# Solve problem with sketching.
for trial in np.arange(0, trials):
    print("Trial: ", trial)
    sjlt_solver =  Logistic_regression_L1_regularization(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, sketch_size1, 3)
    tic2 = time()
    w4, _lambda4, nu4, residual_norms4, times4, alphas4, obj4, rd4, rcent4, rfeas4, residual_opt_conds4 = sjlt_solver.solve(max_iter, w0, _lambda0, nu0, mu) 
    toc2 = time() - tic2
    print("TIme 2:", toc2)
    residual_opt_conds_sketching_instance2.append(residual_opt_conds4)
    times_instance2.append(times4)


# Store data
problem_instance_2_data = {"times_instance2": times_instance2, "residual_opt_conds_sketching_instance2": residual_opt_conds_sketching_instance2,
                           "residual_opt_conds3": residual_opt_conds3, "times3": times3}

with open('plot2_problem_instance_2_data.pickle', 'wb') as handle:
    pickle.dump(problem_instance_2_data, handle)