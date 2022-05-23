# Create plot 2 in my report.
import numpy as np
from matplotlib import pyplot as plt
import pickle

with open('plot1_problem_instance_1_data.pickle', 'rb') as handle:
    data1 = pickle.load(handle)

accuracy_after_60_iter_instance1 = data1["accuracy_after_60_iter_instance1"]
accuracy_after_40_iter_instance1 = data1["accuracy_after_40_iter_instance1"]
accuracy_after_20_iter_instance1 = data1["accuracy_after_20_iter_instance1"]
sketching_sizes1 = data1['sketching_sizes']

with open('plot1_problem_instance_2_data.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
accuracy_after_60_iter_instance2 = data2["accuracy_after_60_iter_instance2"]
accuracy_after_40_iter_instance2 = data2["accuracy_after_40_iter_instance2"]
accuracy_after_20_iter_instance2 = data2["accuracy_after_20_iter_instance2"]
sketching_sizes2 = data2['sketching_sizes2']




std1_instance1 = np.std(accuracy_after_20_iter_instance1, axis = 0)
mean1_instance1 = np.mean(accuracy_after_20_iter_instance1, axis = 0)
std2_instance1 = np.std(accuracy_after_40_iter_instance1, axis = 0)
mean2_instance1 = np.mean(accuracy_after_40_iter_instance1, axis = 0)
std3_instance1 = np.std(accuracy_after_60_iter_instance1, axis = 0)
mean3_instance1 = np.mean(accuracy_after_60_iter_instance1, axis = 0)

std1_instance2 = np.std(accuracy_after_20_iter_instance2, axis = 0)
mean1_instance2 = np.mean(accuracy_after_20_iter_instance2, axis = 0)
std2_instance2 = np.std(accuracy_after_40_iter_instance2, axis = 0)
mean2_instance2 = np.mean(accuracy_after_40_iter_instance2, axis = 0)
std3_instance2 = np.std(accuracy_after_60_iter_instance2, axis = 0)
mean3_instance2 = np.mean(accuracy_after_60_iter_instance2, axis = 0)


log_mean1_instance1 = np.log10(mean1_instance1)
log_std1_instance1 = np.std(log_mean1_instance1)

log_mean2_instance1 = np.log10(mean2_instance1)
log_std2_instance1 = np.std(log_mean2_instance1)


log_mean3_instance1 = np.log10(mean3_instance1)
log_std3_instance1 = np.std(log_mean3_instance1)

## Instance 2
log_mean1_instance2 = np.log10(mean1_instance2)
log_std1_instance2 = np.std(log_mean1_instance2)

log_mean2_instance2 = np.log10(mean2_instance2)
log_std2_instance2 = np.std(log_mean2_instance2)

log_mean3_instance2 = np.log10(mean3_instance2)
log_std3_instance2 = np.std(log_mean3_instance2)


fig, axs = plt.subplots(2, 3)
axs[0, 0].errorbar(sketching_sizes1, log_mean1_instance1, log_std1_instance1, linestyle = 'None', marker='^')
axs[0, 0].set_xscale('log')
axs[0, 1].errorbar(sketching_sizes1, log_mean2_instance1, log_std2_instance1, linestyle = 'None', marker='^')
axs[0, 1].set_xscale('log')
axs[0, 2].errorbar(sketching_sizes1, log_mean3_instance1, log_std3_instance1, linestyle = 'None', marker='^')
axs[0, 2].set_xscale('log')

axs[1, 0].errorbar(sketching_sizes2, log_mean1_instance2, log_std1_instance2, linestyle = 'None', marker='^')
axs[1, 0].set_xscale('log')
axs[1, 1].errorbar(sketching_sizes2, log_mean2_instance2, log_std2_instance2, linestyle = 'None', marker='^')
axs[1, 1].set_xscale('log')
axs[1, 2].errorbar(sketching_sizes2, log_mean3_instance2, log_std3_instance2, linestyle = 'None', marker='^')
axs[1, 2].set_xscale('log')
axs[0, 0].set_ylabel('Log10 of norm of KKT-residual', fontsize = 8)
axs[1, 0].set_ylabel('Log10 of norm of KKT-residual', fontsize = 8)


for ax in axs.flat:
    ax.set(xlabel='Sketch size', ylim = (-9, -2))



fig.set_size_inches(10.5, 5)
plt.savefig("Iteration_vs_sketch_size.png")