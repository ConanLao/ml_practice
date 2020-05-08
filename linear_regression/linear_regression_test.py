from linear_regression import cost_function
import numpy as np
from scipy.io import loadmat
import os

data = loadmat('../Data/ex5data1.mat')
X, y = data['X'], data['y'].ravel()
Xval, yval = data['Xval'], data['yval'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]

# size of the training set
m = y.size

theta = np.array([1, 1])
J, grad = cost_function(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Cost at theta = [1, 1]:\t   %f ' % J)
print('This value should be about 303.993192)\n' % J)

print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))
print(' (this value should be about [-15.303016, 598.250744])\n')