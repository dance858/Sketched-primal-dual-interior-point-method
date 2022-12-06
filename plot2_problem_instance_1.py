from solver import log_reg_ell1_solver, build_log_reg_ell1_problem
import numpy as np
import pickle


# Define some parameters.
d = 100
n = 65536
rho = 0.98
sketch_size1 = int(0.1*n)
#sketch_size1 = 10*d
trials = 15
alpha_backtrack, beta_backtrack = 0.1, 0.5 
A, y, B, b  = build_log_reg_ell1_problem(n, d, rho)
regularization_parameter = 0.0001
max_iter = 50
mu = 10

_lambda0 = np.random.rand(5*d, 1)                       # Multiplier for non-negativity constraints.
nu0 = np.zeros((2*d, 1))                                # Multiplier for equality constraints.
w0 = np.random.rand(5*d, 1)                             # Initial guess. 


# Solve problem without sketching.
no_sketch_solver = log_reg_ell1_solver(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, 0, 1)
w1, _lambda1, nu1, residual_norms_instance1, times1, alphas1, obj1, rd1, rcent1, rfeas1, residual_opt_conds1 = no_sketch_solver.solve(max_iter, w0, _lambda0, nu0, mu)
_x = w1[0:d] - w1[d:2*d] 


residual_opt_conds_sketching_instance1 = []
times_instance1 = []
# Solve problem with sketching.
for trial in np.arange(0, trials):
    print("Trial: ", trial)
    sjlt_solver =  log_reg_ell1_solver(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, sketch_size1, 3)
    w2, _lambda2, nu2, residual_norms2, times2, alphas2, obj2, rd2, rcent2, rfeas2, residual_opt_conds2 = sjlt_solver.solve(max_iter, w0, _lambda0, nu0, mu) 
    residual_opt_conds_sketching_instance1.append(residual_opt_conds2)
    times_instance1.append(times2)


# Store data
problem_instance_1_data = {"times_instance1": times_instance1, "residual_opt_conds_sketching_instance1": residual_opt_conds_sketching_instance1,
                           "residual_opt_conds1": residual_opt_conds1, "times1": times1}


with open('plot2_problem_instance_1_data.pickle', 'wb') as handle:
    pickle.dump(problem_instance_1_data, handle)