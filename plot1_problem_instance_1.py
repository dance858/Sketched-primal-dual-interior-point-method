from solver import log_reg_ell1_solver, build_log_reg_ell1_problem
import numpy as np
import pickle

# Define some parameters.
#d = 40
#n = 65536
d = 10
n = 100
trials = 3
rho = 0.98
sketching_sizes1 = np.int_(np.rint(np.logspace(np.log10(10*d), np.log10(0.5*n), 5)))
alpha_backtrack, beta_backtrack = 0.1, 0.5 
A, y, B, b  = build_log_reg_ell1_problem(n, d, rho)
regularization_parameter = 0.001
max_iter = 60
mu = 10

_lambda0 = np.random.rand(5*d, 1)                       # Multiplier for non-negativity constraints.
nu0 = np.zeros((2*d, 1))                                # Multiplier for equality constraints.
w0 = np.random.rand(5*d, 1)                             # Initial guess. 


accuracy_after_20_iter = np.zeros((trials, len(sketching_sizes1)))
accuracy_after_40_iter = np.zeros((trials, len(sketching_sizes1)))
accuracy_after_60_iter = np.zeros((trials, len(sketching_sizes1)))

# Solve problem without sketching.
no_sketch_solver = log_reg_ell1_solver(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, 0, 1)
w1, _lambda1, nu1, residual_norms1, times1, alphas1, obj1, rd1, rcent1, rfeas1, residual_opt_conds1  = no_sketch_solver.solve(1, w0, _lambda0, nu0, mu)
_x = w1[0:d] - w1[d:2*d] 





# Solve problem with sketching.
counter1 = -1
for sketching_size in sketching_sizes1:
    counter1 += 1
    for trial in np.arange(0, trials):                                             
       countsketch_solver =  log_reg_ell1_solver(B, b, regularization_parameter, A, y, alpha_backtrack, beta_backtrack, sketching_size, 3)
       w2, _lambda2, nu2, residual_norms2, times2, alphas2, obj2, rd2, rcent2, rfeas2, residual_opt_conds2  = countsketch_solver.solve(max_iter, w0, _lambda0, nu0, mu) 
       if len(residual_opt_conds2) >= 60:
           accuracy_after_60_iter[trial, counter1] = residual_opt_conds2[59]
           accuracy_after_40_iter[trial, counter1] = residual_opt_conds2[39]
           accuracy_after_20_iter[trial, counter1] = residual_opt_conds2[19]
        
       elif len(residual_opt_conds2) >= 40:
           accuracy_after_60_iter[trial, counter1] = residual_opt_conds2[39]
           accuracy_after_40_iter[trial, counter1] = residual_opt_conds2[39]
           accuracy_after_20_iter[trial, counter1] = residual_opt_conds2[19]
       else:
           accuracy_after_60_iter[trial, counter1] = residual_opt_conds2[-1]
           accuracy_after_40_iter[trial, counter1] = residual_opt_conds2[-1]
           accuracy_after_20_iter[trial, counter1] = residual_opt_conds2[19]


       
problem_instance_1_data = {"accuracy_after_60_iter_instance1": accuracy_after_60_iter,
                           "accuracy_after_40_iter_instance1": accuracy_after_40_iter,
                           "accuracy_after_20_iter_instance1": accuracy_after_20_iter,
                           "sketching_sizes": sketching_sizes1}

with open('plot1_problem_instance_1_data.pickle', 'wb') as handle:
    pickle.dump(problem_instance_1_data, handle)