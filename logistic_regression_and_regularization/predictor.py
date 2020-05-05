import os
import numpy as np
import logistic_regression as lr
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt('Data/ex2data1.txt', delimiter=',')
X, y = data[:, 0:2], data[:, 2]
m, n = X.shape

# Visualizes data
# pos = y == 1
# neg = y == 0
# plt.plot(X[pos, 0], X[pos, 1], 'ro')
# plt.plot(X[neg, 0], X[neg, 1], 'bo')
# plt.xlabel('Exam 1 score')
# plt.ylabel('Exam 2 score')
# plt.legend(['Admitted', 'Not admitted'])
# plt.show

X = np.concatenate([np.ones((m, 1)), X], axis=1)

theta = np.zeros(n + 1)
print("theta.shape = ", theta.shape)
print("X.shape = ", X.shape)

# Instead of gradient descent use a optimization library.
# set options for optimize.minimize
options = {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(lr.compute_cost_and_grad,
                        theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

prob = lr.sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = lr.predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')