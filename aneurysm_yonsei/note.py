import numpy as np

a = np.array([1,1,2,3])
arr_hist_sd, arr_edges_sd = np.histogram(a, bins = range(np.max(a)+2))
print(arr_hist_sd)