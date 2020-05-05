import numpy as np
import matplotlib.pyplot as plt
import logistic_regression as lr

data = np.loadtxt('Data/ex2data2.txt', delimiter=',')
X, y = data[:, :2], data[:, 2]
m, n = X.shape

# #Visualizes data
# pos = y == 1
# neg = y == 0
# plt.plot(X[pos, 0], X[pos, 1], 'ro')
# plt.plot(X[neg, 0], X[neg, 1], 'bo')
# plt.xlabel('Microchip Test 1')
# plt.ylabel('Microchip Test 2')
# plt.legend(['y = 1', 'y = 0'])
# plt.show()

X = lr.map_feature(X[:, 0], X[:, 1], 6)
print(X[0, :])

