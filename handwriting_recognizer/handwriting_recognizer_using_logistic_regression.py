import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import logistic_regression as lr

# This script trains 10 separate logistic regression classifiers to recognize the handwritten 0,1,2,...,9
# and puts them together to identify what a digit is inside a 20x20 image.

# 20x20 Input Images of Digits
input_layer_size = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)


#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()  # ravel() turns data['y'] from an m * 1 matrix to a vector.

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size

lambda_ = 0.1
num_labels = 10
all_theta = lr.one_vs_all(X, y, num_labels, lambda_)
pred = lr.predict_one_vs_all(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))