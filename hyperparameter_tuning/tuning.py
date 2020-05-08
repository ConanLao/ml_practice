import numpy as np
from linear_regression.linear_regression import gradient_descent
from linear_regression.linear_regression import cost_function
import matplotlib.pyplot as plt


def learning_curve(X, y, X_val, y_val):
    """
    Plots a graph where x-axis is the size of the training set and y-axis is the Training Set Cost and Cross Validation
    Set Cost.
    The model used here is linear regression and the training algorithm is gradient descent.

    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    X_val : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y_val : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).
    """
    m = y.size
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    alpha = 0.001
    num_iters = 100
    lambda_ = 1
    theta_init = np.zeros(X.shape[1])
    for i in range(1, m + 1):
        theta, _ = gradient_descent(X[:i, :], y[:i], theta_init, alpha, num_iters, lambda_)
        error_train[i - 1], _ = cost_function(X[:i, :], y[:i], theta, 0)
        error_val[i - 1], _ = cost_function(X_val, y_val, theta, 0)
    plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
    plt.legend(['Train Set Cost', 'Cross Validation Set Cost'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Cost')
    plt.show()


def map_to_poly(X, degree):
    X_poly = np.zeros((X.shape[0], degree + 1))
    X_poly[:, 0:2] = X[:, 0:2]
    for p in range(2, degree + 1):
        X_poly[:, p] = X[:, 1] ** p
    return X_poly


def normalize(X):
    mean = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return mean, sigma


def poly_degree_curve(X, y, X_val, y_val, degree):
    """
    Maps an array_like input X into a polynomials with a degree of 50 and plots a graph showing the relationship between
    costs and lambda_.
    The model used here is linear regression and the training algorithm is gradient descent.

    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    X_val : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y_val : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    degree : int
        The highest degree of the polynomials.
    """
    m = y.size
    error_train = np.zeros(degree)
    error_val = np.zeros(degree)
    alpha = 0.001
    num_iters = 10000
    lambda_ = 1
    for i in range(1, degree+1):
        theta_init = np.zeros(i + 1)
        X_poly = map_to_poly(X, i)
        mean = np.mean(X_poly[:, 1:i+1], axis=0)
        sigma = np.std(X_poly[:, 1:i+1], axis=0)
        X_poly[:, 1:i+1] = (X_poly[:, 1:i+1] - mean) / sigma
        X_val_poly = map_to_poly(X_val, i)
        X_val_poly[:, 1:i+1] = (X_val_poly[:, 1:i+1] - mean) / sigma
        theta, _ = gradient_descent(X_poly, y, theta_init, alpha, num_iters, lambda_)
        error_train[i - 1], _ = cost_function(X_poly, y, theta, 0)
        error_val[i - 1], _ = cost_function(X_val_poly, y_val, theta, 0)
    plt.plot(np.arange(1, degree + 1), error_train, np.arange(1, degree + 1), error_val, lw = 2)
    plt.legend(['Train Set Cost', 'Cross Validation Set Cost'])
    plt.xlabel('Degree of the polynomial')
    plt.ylabel('Cost')
    plt.show()


def lambda_curve(X, y, X_val, y_val, lambda_values):
    """
    Maps an array_like input X into polynomials of different degrees and plots a graph showing the relationship between
    costs and degrees.
    The model used here is linear regression and the training algorithm is gradient descent.

    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    X_val : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y_val : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    degree : int
        The highest degree of the polynomials.
    """
    degree = 50
    m = y.size
    error_train = np.zeros(len(lambda_values))
    error_val = np.zeros(len(lambda_values))
    alpha = 0.001
    num_iters = 10000
    i = 0;
    for lambda_ in lambda_values:
        theta_init = np.zeros(degree + 1)
        X_poly = map_to_poly(X, degree)
        mean = np.mean(X_poly[:, 1:], axis=0)
        sigma = np.std(X_poly[:, 1:], axis=0)
        X_poly[:, 1:degree+1] = (X_poly[:, 1:degree+1] - mean) / sigma
        X_val_poly = map_to_poly(X_val, degree)
        X_val_poly[:, 1:degree+1] = (X_val_poly[:, 1:degree+1] - mean) / sigma
        theta, _ = gradient_descent(X_poly, y, theta_init, alpha, num_iters, lambda_)
        error_train[i], _ = cost_function(X_poly, y, theta, 0)
        error_val[i], _ = cost_function(X_val_poly, y_val, theta, 0)
        i += 1
    plt.plot(np.arange(1, len(lambda_values) + 1), error_train, np.arange(1, len(lambda_values) + 1), error_val, lw = 2)
    plt.legend(['Train Set Cost', 'Cross Validation Set Cost'])
    plt.xlabel('indexes of lambda_values')
    plt.ylabel('Cost')
    plt.show()