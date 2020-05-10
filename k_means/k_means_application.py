from scipy.io import loadmat
import numpy as np
from k_means.k_means_utils import run_k_means
from k_means.k_means_utils import get_random_initial_centroids

data = loadmat('../Data/ex7data2.mat')
X = data['X']

K = 3   # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
max_iters = 10

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
idx, centroids = run_k_means(X, initial_centroids, max_iters)
# Expect to have
# idx =  [0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 0]
# centroids =  [[1.95399466 5.02557006]
#  [3.04367119 1.01541041]
#  [6.03366736 3.00052511]]
print(idx)
print(centroids)


random_initial_centroids = get_random_initial_centroids(X, K)
idx, centroids = run_k_means(X, random_initial_centroids, max_iters)
print(idx)
print(centroids)
