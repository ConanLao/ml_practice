# used for manipulating directory paths
import os

import numpy as np
import matplotlib.pyplot as plt
import gradient_descent as gd
import utils.data_preprocessing as dp
from linear_regression import gradient_descent

def normalEqn(X, y):
    """

    :param X: array_like
        The dataset of shape (m x n+1).
    :param y: array_like
        The value at each data point. A vector of shape (m, ).
    :return: theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

# >2 features in the input data
data = np.loadtxt('Data/ex1data2.txt', delimiter=',')
X = data[:, 0: -1]
y = data[:, -1]
# Double checks the loaded data.
# print(X[0, 0], X[0, 1])
# print(y[0])
# print(X.shape)
# print(y.shape)

m = X.shape[0] # size of the training set
X, mean, std = dp.feature_normalize(X)
X = np.concatenate([np.ones((m, 1)), X], axis=1)

J = gd.compute_cost(X, y, theta=np.array([0.0, 0.0, 0.0]))

# initialize fitting parameters
theta = np.zeros(3)

# some gradient descent settings
iterations = 1500
alpha = 0.01
theta, J_history = gd.gradient_descent(X ,y, theta, alpha, iterations)
theta_by_normal_eqn = normalEqn(X, y)

# compare the computed thetas by the two different methods
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))
print('Theta found by normal equation: {:.4f}, {:.4f}, {:.4f}'.format(*theta_by_normal_eqn))

# Estimate the price of a 1650 sq-ft, 3 br house
X_test = [1, 1650, 3]
X_test[1:3] = (X_test[1:3] - mean) / std
price = X_test @ theta # You should change this
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))

# Trying different alpha values
# for alpha in [0.1 * ((1.0 / 3.0) ** exp) for exp in range(0, 10)]:
#     theta, J_history = gd.gradient_descent(X, y, theta, alpha, iterations)
#     plt.plot(np.linspace(0, 1500, 15), J_history[0::100], '-', color = 'green')

# alpha = 1 will generate an error because the it's too large that gradient descent doesn't converge

alpha = 0.3
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
plt.plot(np.linspace(0, 1500, 1500), J_history[0::1], '-', color = 'black')
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))

alpha = 0.1
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
plt.plot(np.linspace(0, 1500, 1500), J_history[0::1], '-', color = 'purple')
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))

alpha = 0.03
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
plt.plot(np.linspace(0, 1500, 1500), J_history[0::1], '-', color = 'yellow')
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))

alpha = 0.01
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
plt.plot(np.linspace(0, 1500, 1500), J_history[0::1], '-', color = 'green')
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))

alpha = 0.003
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
plt.plot(np.linspace(0, 1500, 1500), J_history[0::1], '-', color = 'red')
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))

alpha = 0.001
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
plt.plot(np.linspace(0, 1500, 1500), J_history[0::1], '-', color = 'blue')
print('Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))
plt.show()


