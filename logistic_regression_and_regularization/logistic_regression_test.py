import numpy as np
import logistic_regression as lr


def compute_cost_and_grad_test():
    # tests
    data = np.loadtxt('Data/ex2data1.txt', delimiter=',')
    X, y = data[:, 0:2], data[:, 2]
    m, n = X.shape
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    theta = np.zeros(n + 1)

    cost, grad = lr.compute_cost_and_grad(theta, X, y)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx): 0.693\n')

    print('Gradient at initial theta (zeros):')
    print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost, grad = lr.compute_cost_and_grad(test_theta, X, y)

    print('Cost at test theta: {:.3f}'.format(cost))
    print('Expected cost (approx): 0.218\n')

    print('Gradient at test theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')


def compute_cost_and_grad_with_reg_test():
    # Tests for compute_cost_and_grad_with_reg
    data = np.loadtxt('Data/ex2data2.txt', delimiter=',')
    X, y = data[:, :2], data[:, 2]
    m, n = X.shape

    X = lr.map_feature(X[:, 0], X[:, 1], 6)

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    # DO NOT use `lambda` as a variable name in python
    # because it is a python keyword
    lambda_ = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = lr.compute_cost_and_grad_with_reg(initial_theta, X, y, lambda_)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx)       : 0.693\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.0085, 0.0001, 0.0377, 0.0235, 0.0393]\n')


    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones(X.shape[1])
    cost, grad = lr.compute_cost_and_grad_with_reg(test_theta, X, y, 10)

    print('------------------------------\n')
    print('Cost at test theta    : {:.2f}'.format(cost))
    print('Expected cost (approx): 3.16\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[2.6342, 2.3982, 2.4478, 2.3869, 2.4033]')


compute_cost_and_grad_test()
compute_cost_and_grad_with_reg_test()
