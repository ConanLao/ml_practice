import numpy as np


def cost_function(X, y, theta, lambda_=0):
    """
    :param X: array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.
    :param y: array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).
    :param theta: array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).
    :return: array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).
    """
    m = X.shape[0]
    J = np.sum((X.dot(theta) - y) ** 2) / (2 * m) + np.sum(lambda_ / (2 * m) * theta[1:] ** 2)
    grad = 1 / m * X.T @ (X.dot(theta) - y)
    grad[1:] += lambda_ / m * theta[1:]
    return J, grad


def gradient_descent(X, y, theta, alpha, num_iters, lambda_=0.0):
    """
    :param X: array_like
              The dataset of shape (m x n+1).
    :param y: array_like
              A vector of shape (m, ) for the values at a given data point.
    :param theta: array_like
           The linear regression parameters. A vector of shape (n+1, )
    :param alpha: float
           The learning rate for gradient descent.
    :param num_iters: int
           The number of iterations to run gradient descent.
    :return:
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.
    """
    m = X.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        J, grad = cost_function(X, y, theta, lambda_)
        J_history.append(J)
        theta -= alpha * grad

    return theta, J_history
