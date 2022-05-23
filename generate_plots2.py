# Create plot 1 in my report.
import numpy as np
from matplotlib import pyplot as plt
import pickle

with open('plot2_problem_instance_1_data.txt', 'rb') as handle:
    data1 = pickle.load(handle)

times_instance1 = data1['times_instance1']
times1  = data1['times1']
residual_opt_conds1 = data1['residual_opt_conds1']
residual_opt_conds_sketching_instance1 = data1['residual_opt_conds_sketching_instance1']
trials1 = len(residual_opt_conds_sketching_instance1)

fig, axs = plt.subplots(2, 2)

# Upper left corner.
axs[0, 0].semilogy(np.arange(1, len(residual_opt_conds1) + 1), residual_opt_conds1, '-o', label="No sketch")
axs[0, 0].set(ylabel= 'Norm of KKT-residual')
axs[0, 0].semilogy(np.arange(1, len(residual_opt_conds_sketching_instance1[0]) + 1), residual_opt_conds_sketching_instance1[0], color = 'red', label = "Sketch")
for trial in np.arange(1, trials1):
    axs[0, 0].semilogy(np.arange(1, len(residual_opt_conds_sketching_instance1[trial]) + 1), residual_opt_conds_sketching_instance1[trial], color = 'red')

axs[0, 0].legend()

# Upper right corner.
axs[0, 1].semilogy(times1, residual_opt_conds1, '-o', label="No sketch")
axs[0, 1].semilogy(times_instance1[0], residual_opt_conds_sketching_instance1[0], color = 'red', label = "Sketch")
for trial in np.arange(1, trials1):
    axs[0, 1].semilogy(times_instance1[trial], residual_opt_conds_sketching_instance1[trial], color = 'red')

axs[0, 1].legend()


with open('plot2_problem_instance_2_data.pickle', 'rb') as handle:
    data2 = pickle.load(handle)

times_instance2 = data2['times_instance2']
times3  = data2['times3']                              # Change here to time3
residual_opt_conds3 = data2['residual_opt_conds3']     # Change here to 3
residual_opt_conds_sketching_instance2 = data2['residual_opt_conds_sketching_instance2']
trials2 = len(residual_opt_conds_sketching_instance2)

# Lower left corner.
axs[1, 0].semilogy(np.arange(1, len(residual_opt_conds3) + 1), residual_opt_conds3, '-o', label="No sketch")
axs[1, 0].set(ylabel= 'Norm of KKT-residual', xlabel = 'Iteration')
axs[1, 0].semilogy(np.arange(1, len(residual_opt_conds_sketching_instance2[0]) + 1), residual_opt_conds_sketching_instance2[0], color = 'red', label = "Sketch")
for trial in np.arange(1, trials2):
    axs[1, 0].semilogy(np.arange(1, len(residual_opt_conds_sketching_instance2[trial]) + 1), residual_opt_conds_sketching_instance2[trial], color = 'red')

axs[1, 0].legend()


# Lower right corner.
axs[1, 1].semilogy(times3, residual_opt_conds3, '-o', label="No sketch")
axs[1, 1].semilogy(times_instance2[0], residual_opt_conds_sketching_instance2[0], color = 'red', label = "Sketch")
for trial in np.arange(1, trials2):
    axs[1, 1].semilogy(times_instance2[trial], residual_opt_conds_sketching_instance2[trial], color = 'red')

axs[1, 1].set(xlabel='Time (s)')
axs[1, 1].legend()

fig.set_size_inches(10.0, 5)
plt.savefig("ResidualKKT_progress.png")