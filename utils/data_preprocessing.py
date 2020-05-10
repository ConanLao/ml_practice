import numpy as np

def feature_normalize(X, ddof=0):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    ddof : degree of freedom

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    """

    # By default np.mean and np.std computes the mean and std of the flattened array.
    mean = np.mean(X, axis=0);
    std = np.std(X, axis=0, ddof=1)
    X_norm = (X - mean) / std

    # mean and std are used in prediction. Use the mean and std of the training set to "normalize" test set.
    return X_norm, mean, std