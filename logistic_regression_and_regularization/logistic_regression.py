import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
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
    y_hat = sigmoid(X @ theta)
    return - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y_hat) * np.log(1 - y_hat))

def gradient_descent(X, y, theta, alpha, num_iters):
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
        J_history.append(compute_cost(X, y, theta))
        theta = theta - (alpha / m) * X.T @ (X @ theta - y)

    return theta, J_history