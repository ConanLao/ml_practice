import numpy as np

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
    return np.sum((X @ theta - y) ** 2) / (2 * m)

def gradient_descent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        J_history.append(compute_cost(X, y, theta))
        theta = theta - (alpha / m) *  X.T @ (X @ theta - y)

    return theta, J_history