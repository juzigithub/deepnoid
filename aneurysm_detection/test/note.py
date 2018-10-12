import numpy as np


a = np.array([[1,2,3,4],
              [5,6,7,8]])


b = np.repeat(a[:,np.newaxis], 2, 0)
c = np.repeat(a[:,np.newaxis], 2, 1)
print(b)
print(c)