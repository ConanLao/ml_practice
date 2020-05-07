import numpy as np
from scipy import optimize


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    """
        Computes the cost of using theta as the parameter for regularized
    logistic regression and the gradient of the cost w.r.t. to the parameters.

    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is
        the number of features including any intercept.

    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (including intercept).

    y : array_like
        The data labels. A vector with shape (m, ).

    lambda_ : float
        The regularization parameter.

    Returns
    -------
    J : float
        The computed value for the regularized cost function.

    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    """

    m = X.shape[0]
    y_hat = sigmoid(X @ theta.T)
    temp = theta
    temp[0] = 0
    J = (-1 / m) * np.sum(y.dot(np.log(y_hat)) + (1 - y).dot(np.log(1 - y_hat))) + lambda_ / (2 * m) * np.sum(theta[1:] ** 2)
    grad = (1 / m) * (X.T @ (y_hat - y)) + lambda_ / m * temp
    return J, grad


def one_vs_all(X, y, num_labels, lambda_):
    """
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n). m is the number of
        data points, and n is the number of features.

    y : array_like
        The data labels. A vector of shape (m, ).

    num_labels : int
        Number of possible labels.

    lambda_ : float
        The logistic regularization parameter.

    Returns
    -------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        (ie. `numlabels`) and n is number of features without the bias.
    """
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for digit in range(0, num_labels):
        y_temp = y == digit
        theta = np.zeros(n + 1)
        options = {'maxiter': 200}
        res = optimize.minimize(compute_cost_and_grad_with_reg,
                            theta,
                            (X, y_temp, lambda_),
                            jac=True,
                            method='CG',
                            options=options)
        theta = res.x
        all_theta[digit, :] = theta

    return all_theta


def predict_one_vs_all(all_theta, X):
    """
        Return a vector of predictions for each example in the matrix X.
        Note that X contains the examples in rows. all_theta is a matrix where
        the i-th row is a trained logistic regression theta vector for the
        i-th class. You should set p to a vector of values from 0..K-1
        (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

        Parameters
        ----------
        all_theta : array_like
            The trained parameters for logistic regression for each class.
            This is a matrix of shape (K x n+1) where K is number of classes
            and n is number of features without the bias.

        X : array_like
            Data points to predict their labels. This is a matrix of shape
            (m x n) where m is number of data points to predict, and n is number
            of features without the bias term. Note we add the bias term for X in
            this function.

        Returns
        -------
        p : array_like
            The predictions for each data point in X. This is a vector of shape (m, ).
    """
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    return np.argmax(sigmoid(all_theta @ X.T), axis = 0)
