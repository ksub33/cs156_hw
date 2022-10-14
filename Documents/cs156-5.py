from cgi import print_arguments
from os import EX_IOERR
from pyexpat.model import XML_CTYPE_ANY
from re import I
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import random
import time

import new_dogshit as nd


N = 100
def pick_point():
  """ returns a random cartesian point """ 
  return np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

def make_line():
  """ makes a line and returns an array of (-b, -m, 1) """
  p1 = pick_point()
  p2 = pick_point()
  m = (p2[1] - p1[1])/(p2[0] - p1[0])
  b = -(p1[1] + m*p1[0]) # negative b
  m = -m
  return np.array([b, m, 1])

def get_w(X, Y):
  X_pinv = np.linalg.pinv(X)
  return np.dot(X_pinv, Y)

def generate_random_data(n):
  """generates 1x3 matrix of x coordinations, where the first column is 
    an artifical value (1) """
  X = np.random.rand(n, 3) * 2 - 1
  X[:, 0] = 1 # fake column
  return X


def generate_non_linear(n):
  X = np.random.rand(n, 3) * 2 - 1
  X[:, 0] = 1
  Y = np.zeros((n, 1)) # same num of rows
  
  for i in range(n):
    Y[i, 0] = np.sign(X[i, 1]**2 + + X[i, 2]**2 - 0.6) 
    if (np.random.randint(1, 10) == 10):
      Y[i, 0] = -Y[i, 0]
    
  return X, Y


def check_above(points, line):
  """ checks if an x point is above or below the line. Line 
      must be of the form (-b, -m, 1)
  """
  Y = np.empty(len(points))
  for x in range(len(points)):
    if (np.dot(points[x], line) <= 0):
      Y[x] = -1
    else:
      Y[x] = 1
  return Y

def train_lin(n=None, X=None, Y=None):
  f = make_line()
  X = generate_random_data(n) if X is None else X
  Y = check_above(X, f) if Y is None else Y
  # print(Y)
  X_pinv = np.linalg.pinv(X)
  w = np.dot(X_pinv, Y)
  # print(np.shape(X_pinv))
  # print(np.shape(Y))
  return f, X, Y, w

# computes the in sample error
def get_E_in(X, Y_in, new_w, n):
  """
  :param f: target function 
  """
  eval = np.dot(X, new_w)
  
  eval[eval >= 0] = 1
  eval[eval < 0] = -1

  # print(np.shape(eval))
  # print(np.shape(Y_in))
  sum_correct = sum(eval == Y_in)

  # returning the fraction of (in sample) points classified incorrectly
  return 1.0 - sum_correct/n


# computes the out of sample rror
def get_E_out(f, w):
  n = 1000
  X_new = generate_random_data(n)

  Y_new = np.dot(X_new, f)

  # binary classification
  Y_new[Y_new >= 0] = 1
  Y_new[Y_new < 0] = -1
  Y_new = np.int8(Y_new)

  eval = np.dot(X_new, w)

  eval[eval >= 0] = 1
  eval[eval < 0] = -1

  sum_correct = sum(eval == Y_new)
  return 1.0 - sum_correct * 1.0/1000

# runs the expirement and returns E in and E out as a tuple
def run_experiment(n):
  f, X, Y, new_w = train_lin(n)
  return new_w, X, Y

def run_and_get_errors(n):
  f, X, Y, new_w = train_lin(n)
  E_in = get_E_in(X, Y, new_w, n)
  E_out = get_E_out(f, new_w)
  return E_in, E_out, new_w, X, Y
  
def non_linear_main():
  x_data, y_data = generate_non_linear(1000)
  x_trans = np.zeros((1000, 6))

  f, X, Y, w = train_lin(X=x_data, Y=y_data)
  E_in = get_E_in(X, Y, w, 1000)
  return E_in

def calc_agree(x_trans, w):
  a_vec = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
  b_vec = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
  c_vec = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
  d_vec = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
  e_vec = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])


  # maybe do h_out = np.sign(np.dot(x, w)) and that does all this
  # print(np.shape(w))
  # print(np.shape(x_trans))
  h_out = np.sign(np.dot(x_trans, w))
  h_out = np.array(h_out[:, 0])

  a_out = np.sign(np.dot(x_trans, a_vec))
  b_out = np.sign(np.dot(x_trans, b_vec))
  c_out = np.sign(np.dot(x_trans, c_vec))
  d_out = np.sign(np.dot(x_trans, d_vec))
  e_out = np.sign(np.dot(x_trans, e_vec))


  a_agree = sum(h_out == a_out)/1000
  b_agree = np.sum(h_out == b_out)/1000
  c_agree = np.sum(h_out == c_out)/1000
  d_agree = np.sum(h_out == d_out)/1000
  e_agree = np.sum(h_out == e_out)/1000

  return [a_agree, b_agree, c_agree, d_agree, e_agree]

def nonlin_trans_lr() :
  x_data, y_data = generate_non_linear(1000)
  x_trans = np.zeros((1000, 6))
  for i in range(1000):
    x_trans[i, 0:2] = x_data[i, 0:2]
    x_trans[i, 3] = x_data[i, 1]*x_data[i, 2]
    x_trans[i, 4] = x_data[i, 1]**2
    x_trans[i, 5] = x_data[i, 2]**2
    if(np.random.randint(1, 10) == 10):
      Y[i, 0] = -Y[i, 0]

  f, X, Y, w = train_lin(X=x_trans, Y=y_data)
  return X, Y, w
  
def get_nonlin_Eout(w, Y):
  X_out, Y = generate_non_linear(1000)
  x_trans = np.zeros((1000, 6))

  for i in range(1000):
    x_trans[i, 0:2] = X_out[i, 0:2]
    x_trans[i, 3] = X_out[i, 1]*X_out[i, 2]
    x_trans[i, 4] = X_out[i, 1]**2
    x_trans[i, 5] = X_out[i, 2]**2
    if(np.random.randint(1, 10) == 10):
      Y[i, 0] = -Y[i, 0]
  
  h_out = np.sign(np.dot(x_trans, w))
  h_out = np.array(h_out[:, 0])
  Y = np.array(Y[:, 0])



  sum_correct = sum(h_out == Y)
  # print(sum_correct)
  return 1 - sum_correct/1000


if __name__ == "__main__":
  E_in_arr = np.empty(1000)
  E_out_arr = np.empty(1000)
  ctr_arr = []
  nonlin_Ein = []
  agree_arr = []
  nonlin_Eout_arr = []

  for x in range(1000):
    e_in, e_out, w, X, Y = run_and_get_errors(100)
    E_in_arr[x] = e_in
    E_out_arr[x] = e_out

  for y in range(1000):
    new_w, X, Y = run_experiment(10)
    ctr, check = nd.do_exper(10, new_w, X, Y)
    ctr_arr.append(ctr)

  for z in range(1000):
    E_in = non_linear_main()
    nonlin_Ein.append(E_in)

  for k in range(1000):
    X_trans, Y, w = nonlin_trans_lr()
    E_out = get_nonlin_Eout(w, Y)
    agree = calc_agree(X_trans, w)

    agree_arr.append(agree)
    nonlin_Eout_arr.append(E_out)
 
  agree_arr = np.array(agree_arr)
  sum_agree = []

  for i in range(5):
    avg_axis = np.average(agree_arr[:, i])
    sum_agree.append(avg_axis)

  total_E_in = sum(np.absolute(E_in_arr))/1000
  total_E_out = sum(np.absolute(E_out_arr))/1000
  ctr_avg = np.average(ctr_arr)
  nonlin_Ein_avg = np.average(nonlin_Ein)
  nonlin_Eout_avg = np.average(nonlin_Eout_arr)

  print("E_in avg ..... :", total_E_in)
  print("E_out avg ..... :", total_E_out)
  print(f"Average iterations for PLA to converge: {ctr_avg}")
  print(f"average of nonlinear {nonlin_Ein_avg}")
  print(f"probablity of weight vector agreeing with h...")
  print(f"a:{sum_agree[0]}, b:{sum_agree[1]}, c:{sum_agree[2]}, d:{sum_agree[3]}, e:{sum_agree[4]}")
  print(f"Average of E_out for nonlinear transform {nonlin_Eout_avg}")