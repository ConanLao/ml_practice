# used for manipulating directory paths
import os

import numpy as np
import matplotlib.pyplot as plt
import gradient_descent as gd


data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
plt.plot(X, y, 'ro', ms=10, mec='k')
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
# plt.show()

m = X.shape[0] # size of the training set
X = np.stack([np.ones(m), X], axis=1)

J = gd.compute_cost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')


# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01
theta, J_history = gd.gradient_descent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

y_predict = X @ theta
plt.plot(X[:, 1], y_predict, '-')
plt.legend(['Training data', 'Linear regression'])
# plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))