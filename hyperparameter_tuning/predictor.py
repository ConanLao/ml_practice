import numpy as np
from scipy.io import loadmat
from hyperparameter_tuning.tuning import learning_curve
from hyperparameter_tuning.tuning import poly_degree_curve
from hyperparameter_tuning.tuning import lambda_curve

# The code in this class predicts the amount of water flowing out of a dam using the change of water level in a
# reservoir.
data = loadmat('../Data/ex5data1.mat')
X, y = data['X'], data['y'].ravel()
X_val, y_val = data['Xval'], data['yval'][:, 0]
X_test, y_test = data['Xtest'], data['ytest'][:, 0]

# size of the training set
m = y.size

# Plot training data
# plt.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

X = np.concatenate([np.ones((m, 1)), X], axis=1)
X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
learning_curve(X, y, X_val, y_val)
poly_degree_curve(X, y, X_val, y_val, 50)
lambda_curve(X, y, X_val, y_val, [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])

