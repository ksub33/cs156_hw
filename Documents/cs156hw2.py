from re import I
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def coinFlips(num_flips, times):
  exp_arr = np.empty([times, num_flips])
  for x in range(0, times, 1):
    flips = np.random.randint(0, 2, size=num_flips)
    exp_arr[x] = flips
  return exp_arr

def getmin(arr):
  min = 100
  ind = -1
  for x in range(np.size(arr, 0)):
    sum = np.sum(arr[x])
  
    if (sum < min):
      min = sum
      ind = x
  
  return ind

total_size = 1000
v1_arr = np.empty(total_size)
vrand_arr = np.empty(total_size)
vmin_arr = np.empty(total_size)

for x in range(total_size) :
  flips_arr = coinFlips(10, 1000)

  min_ind = getmin(flips_arr)
  v1 = (np.sum(flips_arr[0]))/10
  vrand = (np.sum(flips_arr[np.random.randint(0, 1000)]))/10
  vmin = (np.sum(flips_arr[min_ind]))/10

  v1_arr[x] = v1
  vrand_arr[x] = vrand
  vmin_arr[x] = vmin
  
  print(x)

fig, axs = plt.subplots(3)
x = np.arange(0, total_size, 1)

v1_avg = np.sum(v1_arr)/total_size
vrand_avg = np.sum(vrand_arr)/total_size
vmin_avg = np.sum(vmin_arr)/total_size

print(f"v_1 is {v1_avg}")
print(f"v rand is {vrand_avg}")
print(f"v min is {vmin_avg}")

axs[0].plot(x, v1_arr)
axs[1].plot(x, vrand_arr)
axs[2].plot(x, vmin_arr)

plt.show()

print("done")

