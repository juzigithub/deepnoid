import numpy as np

a = np.arange(0, 10, 1).reshape((2,5))
print(a)
b = np.arange(0, 1, 0.1).reshape((2,5))
print(b.shape)
# ab, ba = np.meshgrid(a, b)
# print(ab.flatten())
# print(ba.flatten())


c = np.stack([a,b], axis=2).reshape((-1, 2))
print(c)

