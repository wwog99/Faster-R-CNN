import numpy as np
import torch


x = np.ones([2,2,3])
print(x)
x[:,:,1] = np.array([[2,2],[2,2]])
x[:,:,2] = np.array([[3,3],[3,3]])
print(x)
print(x.shape)


x = x.transpose([2,0,1])
print(x)
print(x.shape)