import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
from scipy.io import loadmat

data = loadmat('../Data/ex8_movies.mat')
Y, R = data['Y'], data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of
# 1682 movies on 943 users

# R is a 1682x943 matrix, where R(i,j) = 1
# if and only if user j gave a rating to movie i

plt.figure(figsize=(8, 8))
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.grid(False)
plt.show()