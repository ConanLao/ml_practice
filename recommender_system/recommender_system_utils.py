import numpy as np


def cost_function(params, Y, R, num_users, num_movies,
                 num_features, lambda_=0.0):
    """
    Collaborative filtering cost function.

    Parameters
    ----------
    params : array_like
        The parameters which will be optimized. This is a one
        dimensional vector of shape (num_movies x num_users, 1). It is the
        concatenation of the feature vectors X and parameters Theta.

    Y : array_like
        A matrix of shape (num_movies x num_users) of user ratings of movies.

    R : array_like
        A (num_movies x num_users) matrix, where R[i, j] = 1 if the
        i-th movie was rated by the j-th user.

    num_users : int
        Total number of users.

    num_movies : int
        Total number of movies.

    num_features : int
        Number of features to learn.

    lambda_ : float, optional
        The regularization coefficient.

    Returns
    -------
    J : float
        The value of the cost function at the given params.

    grad : array_like
        The gradient vector of the cost function at the given params.
        grad has a shape (num_movies x num_users, 1)
    """
    # Unfold the U and W matrices from params
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    theta = params[num_movies * num_features:].reshape(num_users, num_features)

    # J = (1 / 2) * np.sum((theta @ X.T - Y.T) ** 2 * R.T) + lambda_ / 2 * np.sum(theta ** 2) + lambda_ / 2 * np.sum(X ** 2)
    error = (X @ theta.T - Y) * R
    J = (1 / 2) * np.sum(error ** 2) + lambda_ / 2 * np.sum(theta ** 2) + lambda_ / 2 * np.sum(X ** 2)
    theta_grad = np.zeros((num_users, num_features))
    X_grad = np.zeros((num_movies, num_features))
    print('num_users', num_users)
    print('num_features', num_features)
    print('num_movies', num_movies)
    print('error.shape', error.shape)
    print('X.shape', X.shape)

    # alternative solution without vectorization
    # for i in range(0, num_users):
    #     theta_grad[i, :] = X.T @ error[:, i] + lambda_ * theta[i]
    # for j in range(0, num_movies):
    #     X_grad[j, :] = theta.T @ error[j, :].T + lambda_ * X[j]
    theta_grad = (X.T @ error).T + lambda_ * theta
    X_grad = (theta.T @ error.T).T + lambda_ * X

    grad = np.concatenate([X_grad.ravel(), theta_grad.ravel()])
    return J, grad