import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import neural_networks_utils as nnu

data = loadmat(os.path.join('Data', 'ex4data1.mat'))
X, y = data['X'], data['y'].ravel()

y[y == 10] = 0

m = y.shape[0]

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

print('Initializing Neural Network Parameters ...')

theta1_init = nnu.random_init(input_layer_size, hidden_layer_size)
theta2_init = nnu.random_init(hidden_layer_size, num_labels)
nn_param_init = np.concatenate([theta1_init.ravel(), theta2_init.ravel()])

options= {'maxiter': 1000}
lambda_ = 1.0
cost_function_ = lambda p: nnu.cost_function(p, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_)
res = optimize.minimize(cost_function_,
                        nn_param_init,
                        jac=True,
                        method='TNC',
                        options=options)
nn_params = res.x
theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

pred = nnu.predict(theta1, theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y)))