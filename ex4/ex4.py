
# coding: utf-8

# # 1. Import images

# In[1]:

import scipy.io
data = scipy.io.loadmat('data/ex3data1.mat')

# A 5000x400matrix, each row representing a 20x20 pixel image
X = data['X']

# A 5000-dimensional matrix, each element representing the number shown in the corresponding image
Y = data['y'].reshape(5000)


# # 2. Display selection of images

# In[2]:

import numpy as np
from matplotlib import pyplot as plt

# Generating new random numbers each time I executed the kernel meant that
# a new image had to be generated, so I wrote a script to generate a
# hundred random numbers between 0 and 4999 and copied the output here
rand_nums = np.array([457, 4925, 31, 3479, 4392, 4110, 107, 4292, 1345, 4614,
              2150, 4885, 4793, 3883, 4483, 2977, 1859, 1417, 4038, 957,
              430, 3598, 300, 431, 4691, 1799, 4190, 3643, 1084, 4735,
              2327, 4340, 4743, 4754, 277, 244, 2213, 2740, 3237, 3709,
              3976, 2278, 87, 539, 2720, 683, 4561, 481, 3181, 4488, 1201,
              47, 1028, 2608, 2036, 92, 332, 1804, 3586, 4337, 2778, 1057,
              3827, 2949, 1906, 2751, 3865, 4615, 2612, 4600, 503, 4705,
              4014, 3409, 618, 1735, 1587, 1, 4115, 4910, 1780, 4219, 2058,
              3611, 4372, 1424, 996, 821, 289, 492, 954, 4034, 1789, 2137,
              3246, 1474, 1038, 3524, 4456, 4093])

img_arr = np.empty([20, 220])
for j in range(0, 10):
    # Array to hold a row of 10 images
    img_row = np.empty([20, 20])
    for i in range(0, 10):
        img = X[rand_nums[(10*j) + i]].reshape(20, 20)
        img_row = np.concatenate((img_row, img), axis=1)
    # Add row to the image matrix
    img_arr = np.concatenate((img_arr, img_row), axis=0)

plt.figure(figsize=(12,8))
plt.imshow(img_arr[20:, 20: ].T, interpolation="nearest", cmap='gray')

# Show random selection of 100 images using imshow
plt.show()


# # 3. Cost function

# ## 3.1 Actual Definition

# In[3]:

from scipy.special import expit
import math

def h(theta, vec):
    return expit(np.dot(vec, theta))

def cost_function(theta1, theta2, X, Y, K):
    m = Y.shape[0]
    cost = 0
    # For every training example:
    for i in range(0, m):

        # Hidden layer
        z2 = np.dot(theta1, X[i].T)
        a2 = expit(z2)
        # Add bias nodes
        a2 = np.concatenate((np.ones(1), a2), axis=0)

        # Final layer
        z3 = np.dot(theta2, a2.T)
        a3 = expit(z3)
        hypothesis = a3

        # construct y
        y = np.zeros(10)
        # Suppose Y[i] = 5, y needs to be a 10-dimensional matrix with y[4] = 1 (becoz indexing starts with zero)
        y[Y[i] - 1] = 1

        #  For every class:
        for k in range(0, K):
            first_term = -y[k] * math.log(hypothesis[k])
            second_term = -(1 - y[k]) * math.log(1 - hypothesis[k])
            cost = cost + (first_term + second_term)

    return cost/m


# ## 3.2 Testing the cost function against given theta

# In[4]:

data_nn = scipy.io.loadmat('data/ex3weights.mat')
theta1 = data_nn['Theta1']
theta2 = data_nn['Theta2']

X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

print "Testing cost function with given weights:",cost_function(theta1, theta2, X, Y, 10)


# ## 3.3 Regularized cost function

# In[5]:

def regularized_cost_function(theta1, theta2, X, Y, K, lambda_):
    m = Y.shape[0]
    cost = cost_function(theta1, theta2, X, Y, K)
    regularization_term = 0
    for j in range(0, theta1.shape[0]):
        for k in range(1, theta1.shape[1]):
            regularization_term = regularization_term + (theta1[j, k]*theta1[j, k])

    for j in range(0, theta2.shape[0]):
        for k in range(1, theta2.shape[1]):
            regularization_term = regularization_term + (theta2[j, k]*theta2[j, k])

    regularization_term = (regularization_term*lambda_) / (2*m)
    return cost + regularization_term

print "Regularized Cost:", regularized_cost_function(theta1, theta2, X, Y, 10, 1)

