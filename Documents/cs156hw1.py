from re import I
import re
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def pick_point():
  return np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

def make_line():
  p1 = pick_point()
  p2 = pick_point()
  m = (p2[1] - p1[1])/(p2[0] - p1[0])
  b = (p1[1] - m*p1[0])
 
  return np.array([1, m, b])

def generate_random_data(n):
  X = np.random.rand(n, 3) * 2 - 1
  X[:, 0] = 1 # fake column
  return X
# return the sum of x_o * w_o + x_1 * w_1 + .... + x_n * w_n

def test(X, line):
  Y = np.dot(X, line)
  Y[Y >= 0] = 1
  Y[Y < 0] = -1
  return Y

def test_0(points, line):
  above = []
  below = []
  for x in points:
    if np.dot(line, x) >= 0:
      above.append(x)
    else:
      below.append(x)
  return np.asarray(above), np.asarray(below)

def h_x(w, x):
  # both should have three elements
  return np.dot(w, x)

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

X  = generate_random_data(10)
line = make_line()
abline(line[1], line[2])
Y = test(X, line)
X = np.delete(X, 0, 1)
# print(X)
print(Y)
x_1 = X[0, :]
# print(x_1)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

