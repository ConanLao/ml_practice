import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
from scipy.io import loadmat
from recommender_system_utils import cost_function
from recommender_system_utils import normalize_ratings

data = loadmat('../Data/ex8_movies.mat')
Y, R = data['Y'], data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of
# 1682 movies on 943 users

# R is a 1682x943 matrix, where R(i,j) = 1
# if and only if user j gave a rating to movie i

#  Load data
data = loadmat('../Data/ex8_movies.mat')
Y, R = data['Y'], data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i


with open('../Data/movie_ids.txt', encoding='ISO-8859-1') as f:
    content = f.readlines(-1)
movie_ids = [x.strip() for x in content]
print(movie_ids)

my_ratings = np.zeros(len(movie_ids))
my_ratings[63] = 5 # 64 Shawshank Redemption, The (1994)
my_ratings[70] = 5 # 71 Lion King, The (1994)
my_ratings[68] = 5 # 69 Forrest Gump (1994)
my_ratings[132] = 5 # 133 Gone with the Wind (1939)
my_ratings[88] = 5 # 89 Blade Runner (1982)
my_ratings[177] = 5 # 178 12 Angry Men (1957)

print('My ratings:')
print('-----------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d stars: %s' % (my_ratings[i], movie_ids[i]))

Y = np.concatenate([my_ratings[:, None], Y], axis=1)
R = np.concatenate([(my_ratings > 0)[:, None], R], axis=1)

#  Normalize Ratings
Ynorm, Ymean = normalize_ratings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate([X.ravel(), Theta.ravel()])

# Train the model
options = {'maxiter': 100}
lambda_ = 10
res = optimize.minimize(lambda x: cost_function(x, Ynorm, R, num_users,
                                               num_movies, num_features, lambda_),
                        initial_parameters,
                        method='TNC',
                        jac=True,
                        options=options)
theta = res.x

# Unfold the returned theta back into U and W
X = theta[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

print('Recommender system learning completed.')

p = np.dot(X, Theta.T)
my_predictions = p[:, 0] + Ymean

ix = np.argsort(my_predictions)[::-1]

print('Top recommendations for you:')
print('----------------------------')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j], movie_ids[j]))

print('\nOriginal ratings provided:')
print('--------------------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movie_ids[i]))