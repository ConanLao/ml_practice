import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost_and_grad(theta, X, y):
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

        grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    """
    m = X.shape[0]
    y_hat = sigmoid(X @ theta.T)
    J = - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    grad = (1 / m) * X.T @ (y_hat - y)
    return J, grad


def predict(theta, X):
    return sigmoid(X @ theta) > 0.5


def map_feature(X1, X2, degree = 6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    m = X1.shape[0]
    # number of columns(features) = 1 + 2 + 3 + ... + (degree + 1)
    result = np.zeros((m, int((1 + degree + 1) * (degree + 1) / 2)))
    index = 0
    for i in range(0, degree + 1):
        for j in range(0, degree - i + 1):
            result[:, index] = X1 ** i * X2 ** j
            index += 1
    return result


def compute_cost_and_grad_with_reg(theta, X, y, lambda_):
    m = X.shape[0]
    y_hat = sigmoid(X @ theta.T)
    J = - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + np.sum(lambda_ / (2 * m) * theta[1:] ** 2)
    grad = (1 / m) * X.T @ (y_hat - y) + lambda_ / m * np.sum(theta[1:])
    return J, grad
