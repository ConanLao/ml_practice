from scipy.io import loadmat
from utils.data_preprocessing import feature_normalize
from utils.pca_utils import pca
from utils.pca_utils import project_data
from utils.pca_utils import recover_data

data = loadmat('../Data/ex7data1.mat')
X = data['X']

X_norm, mu, sigma = feature_normalize(X)

#  Run PCA
U, S = pca(X_norm)
K = 1
Z = project_data(X_norm, U, K)

print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
print('(this value should be about    : 1.481274)')


X_rec  = recover_data(Z, U, K)
print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
print('       (this value should be about  [-1.047419 -1.047419])')