from solver import log_reg_ell1_solver, build_log_reg_ell1_problem
import numpy as np
import pickle
from time import time


# Define some parameters.
d = 10
n = 1000
rho = 0.98
sketch_size1 = int(0.1 * n)             
#sketch_size1 = 3000
trials = 1
alpha_backtrack, beta_backtrack = 0.1, 0.5 
A, y, B, b  = build_log_reg_ell1_problem(n, d, rho)
regularization_parameter = 0.0001
max_iter = 50
mu = 10

_lambda0 = np.random.rand(5*d, 1)                       # Multiplier for non-negativity constraints.
nu0 = np.zeros((2*d, 1))                                # Multiplier for equality constraints.
w0 = np.random.rand(5*d, 1)                             # Initial guess. 


# Solve problem without sketching.
no_sketch_solver = log_reg_ell1_solver(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, 0, 1, False)
tic1 = time()
w3, _lambda3, nu3, residual_norms_instance3, times3, alphas3, obj3, rd3, rcent3, rfeas3, residual_opt_conds3 = no_sketch_solver.solve(2, w0, _lambda0, nu0, mu)
toc1 = time() - tic1
print("Time 1:", toc1)
_x = w3[0:d] - w3[d:2*d]
residual_opt_conds_sketching_instance2 = []
times_instance2 = []
# Solve problem with sketching.
for trial in np.arange(0, trials):
    print("Trial: ", trial)
    sjlt_solver =  log_reg_ell1_solver(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, sketch_size1, 3, True)
    tic2 = time()
    w4, _lambda4, nu4, residual_norms4, times4, alphas4, obj4, rd4, rcent4, rfeas4, residual_opt_conds4 = sjlt_solver.solve(max_iter, w0, _lambda0, nu0, mu) 
    toc2 = time() - tic2
    print("TIme 2:", toc2)
    residual_opt_conds_sketching_instance2.append(residual_opt_conds4)
    times_instance2.append(times4)




# Store data
problem_instance_2_data = {"times_instance2": times_instance2, "residual_opt_conds_sketching_instance2": residual_opt_conds_sketching_instance2,
                           "residual_opt_conds3": residual_opt_conds3, "times3": times3}

#problem_instance_2_data_new = {"primal_feas_non_sketched": rfeas3, "dual_feas_non_sketched": rd3, "cent_non_sketched_feas": rcent3,
#                               "primal_feas_sketched": rfeas4, "dual_feas_ketched": rd4, "cent_sketched_feas": rcent4}

with open('plot2_problem_instance_2_data_correction.pickle', 'wb') as handle:
    pickle.dump(problem_instance_2_data, handle)

#with open('data_residual_analysis.pickle', 'wb') as handle:
#    pickle.dump(problem_instance_2_data_new, handle)