import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(theta1, theta2, X):
    m = X.shape[0]
    a1 = np.concatenate([np.ones((m, 1)), X], axis = 1).T
    z2 = theta1 @ a1
    a2 = sigmoid(z2)
    a2 = np.concatenate([np.ones((1, m)), a2], axis = 0)
    z3 = theta2 @ a2
    a3 = sigmoid(z3)
    return np.argmax(a3, axis = 0)

# def predict(Theta1, Theta2, X):
#     """
#     Predict the label of an input given a trained neural network
#     Outputs the predicted label of X given the trained weights of a neural
#     network(Theta1, Theta2)
#     """
#     # Useful values
#     m = X.shape[0]
#     num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p

def cost_function(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    """
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.

    input_layer_size : int
        Number of features for the input layer.

    hidden_layer_size : int
        Number of hidden units in the second layer.

    num_labels : int
        Total number of labels, or equivalently number of units in output layer.

    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).

    y : array_like
        Dataset labels. A vector of shape (m,).

    lambda_ : float, optional
        Regularization parameter.

    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.

    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenation of
        neural network weights Theta1 and Theta2.
    """

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size+1))
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
    m = y.shape[0]

    # forward
    # all a's and z's have shape {nodes in the layer + 1} * m
    a1 = np.concatenate([np.ones((m, 1)), X], axis = 1).T
    z2 = theta1 @ a1
    a2 = sigmoid(z2)
    a2 = np.concatenate([np.ones((1, m)), a2], axis = 0)
    z3 = theta2 @ a2
    a3 = sigmoid(z3)
    # Turns a m * 1 matrix into a vector
    y = y.reshape(-1)
    # Gets a num_labels * m result matrix
    y = np.eye(num_labels)[y].T
    J = -1 / m * np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
    J += lambda_ / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

    # backward
    d3 = a3 - y
    d2 = theta2[:, 1:].T @ d3 * sigmoid_gradient(z2)

    delta2 = d3 @ a2.T
    delta1 = d2 @ a1.T

    # computes theta's
    grad_theta1 = (1 / m) * delta1
    grad_theta1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    grad_theta2 = (1 / m) * delta2
    grad_theta2[:, 1:] += (lambda_ / m) * theta2[:, 1:]
    grad = np.concatenate([grad_theta1.ravel(), grad_theta2.ravel()])

    return J, grad

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def random_init(L_in, L_out, epsilon_init=0.12):
    """
    Parameters
    ----------
    L_in : int
        Number of incomming connections.

    L_out : int
        Number of outgoing connections.

    epsilon_init : float, optional
        Range of values which the weight can take from a uniform
        distribution.

    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
    """
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def computeNumericalGradient(theta, cost_function, e=1e-4):
    grad = np.zeros(theta.size)
    ex2 = e * 2;
    for i in range(0, theta.size):
        theta_left = theta.copy();
        theta_left[i] -= e;
        loss_left, _ = cost_function(theta_left)
        theta_right = theta.copy();
        theta_right[i] += e;
        loss_right, _ = cost_function(theta_right)
        grad[i] = (loss_right - loss_left) / ex2
    return grad


def check_gradients(cost_function):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    lambda_ = 3
    theta1 = random_init(hidden_layer_size, input_layer_size)
    theta2 = random_init(num_labels, hidden_layer_size)
    X = np.random.rand(m, input_layer_size)
    y = np.arange(1, 1+m) % num_labels
    nn_params = np.concatenate([theta1.ravel(), theta2.ravel()])

    c_function = lambda p: cost_function(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lambda_)

    cost, grad = c_function(nn_params)
    numgrad = computeNumericalGradient(nn_params, c_function)

    # Visually examine the two gradient computations.The two columns you get should be very similar.

    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)