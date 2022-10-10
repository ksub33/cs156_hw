from re import I
import matplotlib.pyplot as plt
import numpy as np
import random
import time

N = 100

def pick_point():
  return np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

def make_line()pick_point:
  p1 = ()
  p2 = pick_point()
  m = (p2[1] - p1[1])/(p2[0] - p1[0])
  b = (p1[1] - m*p1[0])

  # this need to have 1 because we're going to dot it with x which has 1
  return np.array([1, m, b])

def get_w(X, Y):
  X_pinv = np.linalg.pinv(X)
  return np.dot(X_pinv, Y)

def generate_rand_data(n):
  # basically we make some line, and then generate a bunch of random x coords 
  # then we get an target data vector of Y and then we make it so that
  # we only know if the y is above or below the line

  # input data matrix
  w = make_line()

  # n * 3 matrix, have two x entries because d = 2, first entry is classification
  X = np.random.rand(n, 3) * 2 - 1
  X[:, 0] = 1

  Y = np.dot(X, w)
  
  # binary classification
  Y[Y >= 0] = 1
  Y[Y < 0] = -1

  return w, X, Y

# uses old w, X, and Y and returns g (aka new w)
def train_lin(n):
  w, X, Y = generate_rand_data(n)
  X_pinv = np.linalg.pinv(X)
  new_w = np.dot(X_pinv, Y)
  return w, X, Y, new_w

# computes the in sample error
def get_E_in(w, X, Y_in, new_w):
  eval = np.dot(X, new_w)
  
  eval[eval >= 0] = 1
  eval[eval < 0] = -1

  sum_correct = sum(eval == Y_in)

  # returning the fraction of (in sample) points classified incorrectly
  return 1.0 - sum_correct/N


# computes the out of sample rror
def get_E_out(old_w, new_w):
  n = 1000
  X_new = np.random.rand(n, 3) * 2 - 1
  X_new[:, 0] = 1

  Y_new = np.dot(X_new, old_w)

  # binary classification
  Y_new[Y_new >= 0] = 1
  Y_new[Y_new < 0] = -1
  Y_new = np.int8(Y_new)

  eval = np.dot(X_new, new_w)

  eval[eval >= 0] = 1
  eval[eval < 0] = -1

  sum_correct = sum(eval == Y_new)
  # print("E out ... ", 1.0 - sum_correct * 1.0/1000)
  # returning the fraction of (out of sample) points classified incorrectly
  return 1.0 - sum_correct * 1.0/1000

# runs the expirement and returns E in and E out as a tuple
def run_experiment(n):
  w, X, Y, new_w = train_lin(n)
  E_in = get_E_in(w, X, Y, new_w)
  E_out = get_E_out(w, new_w)
  return E_in, E_out


if __name__ == "__main__":
  E_in_arr = np.empty(1000)
  E_out_arr = np.empty(1000)
  for x in range(1000):
    e_in, e_out = run_experiment(100)
    E_in_arr[x] = e_in
    E_out_arr[x] = e_out

  total_E_in = sum(np.absolute(E_in_arr))/1000
  total_E_out = sum(np.absolute(E_out_arr))/1000

  print("E_in avg ..... ", total_E_in)
  print("E_out avg ..... ", total_E_out)
