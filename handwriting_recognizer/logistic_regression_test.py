import numpy as np
import logistic_regression as lr


def compute_cost_and_grad_with_reg_test():
    theta_t = np.array([-2, -1, 1, 2], dtype=float)

    # test values for the inputs
    X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10.0], axis=1)

    # test values for the labels
    y_t = np.array([1, 0, 1, 0, 1])

    # test value for the regularization parameter
    lambda_t = 3

    J, grad = lr.compute_cost_and_grad_with_reg(theta_t, X_t, y_t, lambda_t)

    print('Cost         : {:.6f}'.format(J))
    print('Expected cost: 2.534819')
    print('-----------------------')
    print('Gradients:')
    print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
    print('Expected gradients:')
    print(' [0.146561, -0.548558, 0.724722, 1.398003]');

compute_cost_and_grad_with_reg_test()