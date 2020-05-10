import os
import numpy as np
import random

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import optimize
from scipy.io import loadmat


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : array_like
        The dataset of size (m, n) where each row is a single example.
        That is, we have m examples each of n dimensions.

    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the the data dimension.

    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset X.
    """
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(0, X.shape[0]):
        idx[i] = np.argmin(np.sum((X[i] - centroids) ** 2, axis=1))

    # alternative implementation
    #
    # min_dist_squares = np.zeros(X.shape[0], dtype=int)
    #
    # for i in range(0, K):
    #     dist_squares = np.sum((X - centroids[i]) ** 2, axis=1)
    #     if i == 0:
    #         min_dist_squares = np.sum((X - centroids[i]) ** 2, axis=1)
    #     else:
    #         idx[dist_squares < min_dist_squares] = i
    #         print(dist_squares.shape)
    #         print(min_dist_squares.shape)
    #         min_dist_squares = np.minimum(min_dist_squares, dist_squares)

    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.

    Parameters
    ----------
    X : array_like
        The datset where each row is a single data point. That is, it
        is a matrix of size (m, n) where there are m datapoints each
        having n dimensions.

    idx : array_like
        A vector (size m) of centroid assignments (i.e. each entry in range [0 ... K-1])
        for each example.

    K : int
        Number of clusters

    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data
        points assigned to it.

    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(0, K):
        closest_to_centroid_i = X[idx == i]
        centroids[i] = np.sum(closest_to_centroid_i, axis = 0) / closest_to_centroid_i.shape[0]
    return centroids


def run_k_means(X, initial_centroids, max_iters):
    centroids = initial_centroids.copy()
    for i in range(0, max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, initial_centroids.shape[0])
    return idx, centroids


def get_random_initial_centroids(X, K):
    """
    :param X:  unlabeled data
    :param K:  number of centroids
    :return: K vectors randomly sampled vectors from X
    """
    random_indices = random.sample(range(0, X.shape[0]), K)
    return X[random_indices]
