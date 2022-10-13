from cgi import print_environ
from re import I, M
import re
from tabnanny import check
from webbrowser import get
import matplotlib.pyplot as plt
import numpy as np
import random
import time
# np.random.seed(29018390)

def pick_point():
  return np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

def make_line():
  p1 = pick_point()
  p2 = pick_point()
  m = (p2[1] - p1[1])/(p2[0] - p1[0])
  b = -(p1[1] + m*p1[0]) # negative b
  m = -m
  return np.array([b, m, 1])

def generate_random_data(n):
  X = np.random.rand(n, 3) * 2 - 1
  X[:, 0] = 1 # fake column
  return X

def check_above(points, line):
  Y = np.empty(len(points))
  for x in range(len(points)):
    if (np.dot(points[x], line) <= 0):
      Y[x] = -1
    else:
      Y[x] = 1
  return Y

def get_missclassed(X, Y, w):
  X_evaled = [np.dot(x, w) for x in X]
  missclassed_ind = -1

  misclassed = []
  for i in range(len(X_evaled)):
    if (np.sign(X_evaled[i]) != Y[i]):
      misclassed.append((X[i], Y[i]))
  
  if (len(misclassed) == 0):
    return weights

  
  rand_missed = random.choice(misclassed)
  yx = rand_missed[1] * rand_missed[0]
  weights = np.add(w, yx)
  return weights

# runs one experiment and returns the amount of iterations it took
#  

def check_model(f, g, n):
  # f and g are weight vectors 
  misclassed = []
  data = generate_random_data(n)
  f_evaled = check_above(data, f)
  g_evaled = [np.dot(g, d_n) for d_n in data]
  for x in range(len(data)):
    if (f_evaled[x] != np.sign(g_evaled[x])):
      misclassed.append((f_evaled[x], g_evaled[x]))

  return len(misclassed)/n

def do_exper(n):
  line = make_line()
  X = generate_random_data(n)
  Y = check_above(X, line)
  weights = np.array([0,0,0])
  ctr = 0
  while(True):
    ctr += 1
    X_evaled = [np.dot(x, weights) for x in X]

    misclassed = []
    misclassed.clear()
    for i in range(len(X_evaled)):
      if (np.sign(X_evaled[i]) != Y[i]):
        misclassed.append((X[i], Y[i]))

    # print(misclassed)
    if (len(misclassed) == 0):
      # print("true..............")
      break

    else:
      # print("yahhhhhhhhhhh")
      rand_missed = random.choice(misclassed)
      yx = rand_missed[1] * rand_missed[0]
      # yx[0] = np.abs(yx[0])
      weights = np.add(weights, yx)
      # print(weights[0])
  # print(f"NUM ITERS = {ctr}")
  check = check_model(line, weights, 10)
  return ctr, check


  # code to plot lines .... doesnt work always 

  # if(weights[0] == 0):
  #   # print("..............weights zero is zero..........................")
  #   weights[0] = 1
  # print(weights)
  # X1 = X[np.dot(X, line) <= 0]
  # X2 = X[np.dot(X, line) > 0]
  
  # plt.scatter(X1[:, 1], X1[:, 2], color="red")
  # plt.scatter(X2[:, 1], X2[:, 2], color="blue")
  

  # # print(weights)
  # abline(-line[1], -line[0])
  # # print(f"w_1:{weights[0]}, w_1:{weights[1]}, w_3:{weights[2]}")
  

  # w_slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
  # w_int = -(weights[0])/weights[2]

  # # print(f"w slope is {w_slope}")
  # # print(f"w_int is {w_int}")
  # abline(w_slope, w_int)

  # # plt.scatter(X[:, 0], X[:, 1])
  # plt.show()

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

if __name__ == "__main__":
  iters = []
  diff = []
  for x in range(1000):
    curr_test = do_exper(10)
    iters.append(curr_test[0])
    diff.append(curr_test[1])
  
  print(39 * "-")
  print(f"N = 10, 1000 iterations \naverage of iters ... {np.average(iters)}")
  print(f"average of diff ... {np.average(diff)} \n")
  print(39 * "-")
  
  iters.clear()
  diff.clear()

  for y in range(1000):
    curr_test = do_exper(100)
    iters.append(curr_test[0])
    diff.append(curr_test[1])
  print(f"N = 100, 1000 iterations\naverage of iters ... {np.average(iters)}")
  print(f"average of diff ... {np.average(diff)} \n")
  print(39 * "-")




# line = make_line()

# def abline(slope, intercept):
#     """Plot a line from slope and intercept"""
#     axes = plt.gca()
#     x_vals = np.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, '--')

# abline(-line[1], -line[0])
# X = generate_random_data(15)
# Y = check_above(X, line)
# print(Y)

# X1 = X[np.dot(X, line) <= 0]
# X2 = X[np.dot(X, line) > 0]

# plt.scatter(X1[:, 1], X1[:, 2], color="red")
# plt.scatter(X2[:, 1], X2[:, 2], color="blue")
# plt.show()
