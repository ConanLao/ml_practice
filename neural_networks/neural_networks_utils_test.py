import neural_networks_utils as nnu
import numpy as np
from scipy.io import loadmat
import os

data = loadmat(os.path.join('Data', 'ex4data1.mat'))
X, y = data['X'], data['y']

y[y == 10] = 0

m = y.shape[0]

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the weights into variables Theta1 and Theta2
weights = loadmat(os.path.join('Data', 'ex4weights.mat'))
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll parameters
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

lambda_ = 1
J, grad = nnu.cost_function(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %.6f' % J)
print('This value should be about                 : 0.383770.')

#  Check gradients by running checkNNGradients
lambda_ = 3

# Also output the costFunction debugging values
debug_J, _  = nnu.cost_function(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
print('(for lambda = 3, this value should be about 0.576051)')

nnu.check_gradients(nnu.cost_function)