# In this file we test the effect of applying the correction step to problem instance 1.
from solver import log_reg_ell1_solver, build_log_reg_ell1_problem
import numpy as np
import pickle
from time import time

# Define some parameters.
d = 40
n = 65336
rho = 0.98
sketch_size1 = int(0.1 * n)             
trials = 5
alpha_backtrack, beta_backtrack = 0.1, 0.5 
A, y, B, b  = build_log_reg_ell1_problem(n, d, rho)
gamma = 0.0001
max_iter = 50
mu = 10

_lambda0 = np.random.rand(5*d, 1)                       # Multiplier for non-negativity constraints.
nu0 = np.zeros((2*d, 1))                                # Multiplier for equality constraints.
w0 = np.random.rand(5*d, 1)                             # Initial guess. 


# Solve problem without sketching.
no_sketch_solver = log_reg_ell1_solver(B=B, b=b, gamma=gamma, A=A, y=y,
                                       alpha_backtrack=alpha_backtrack,
                                       beta_backtrack=beta_backtrack,
                                       sketching_size=0, sketching_strategy=1,
                                       correction=False)
tic1 = time()
w3, _lambda3, nu3, residual_norms_instance3, times3, alphas3, obj3, rd3, rcent3, \
     rfeas3, residual_opt_conds3 = no_sketch_solver.solve(max_iter, w0, _lambda0, nu0, mu)

toc1 = time() - tic1
#print("Time 1:", toc1)
residual_opt_conds_sketching_no_correction_instance2 = []
residual_opt_conds_sketching_correction_instance2 = []
times_instance2 = []
# Solve problem with sketching.
for trial in np.arange(0, trials):
    print("Trial: ", trial)
    countsketch_no_correction =  log_reg_ell1_solver(B, b, gamma, A, y, alpha_backtrack, beta_backtrack, sketch_size1, 3, False)
    countsketch_correction =  log_reg_ell1_solver(B, b, gamma, A, y, alpha_backtrack, beta_backtrack, sketch_size1, 3, True)
    tic2 = time()
    w4, _lambda4, nu4, residual_norms4, times4, alphas4, obj4, rd4, rcent4, rfeas4, residual_opt_conds4 = countsketch_no_correction.solve(max_iter, w0, _lambda0, nu0, mu) 
    w5, _lambda5, nu5, residual_norms5, times5, alphas5, obj5, rd5, rcent5, rfeas5, residual_opt_conds5 = countsketch_correction.solve(max_iter, w0, _lambda0, nu0, mu) 
    toc2 = time() - tic2
    #print("Time 2:", toc2)
    residual_opt_conds_sketching_no_correction_instance2.append(residual_opt_conds4)
    residual_opt_conds_sketching_correction_instance2.append(residual_opt_conds5)
    times_instance2.append(times4)




# Store data
problem_instance_2_data = {"times_instance2": times_instance2, "residual_opt_conds_sketching_no_correction_instance2": residual_opt_conds_sketching_no_correction_instance2,
                           "residual_opt_conds3": residual_opt_conds3, "times3": times3, "residual_opt_conds_sketching_correction_instance2":residual_opt_conds_sketching_correction_instance2}

with open('data/correction_instance_1_data.pickle', 'wb') as handle:
    pickle.dump(problem_instance_2_data, handle)

