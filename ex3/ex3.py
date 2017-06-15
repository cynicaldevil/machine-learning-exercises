
# coding: utf-8

# # 1. Import images

# In[1]:

import scipy.io
data = scipy.io.loadmat('data/ex3data1.mat')

# A 5000x400matrix, each row representing a 20x20 pixel image
X = data['X']

# A 5000-dimensional matrix, each element representing the number shown in the corresponding image
Y = data['y']


# # 2. Display selection of images

# In[2]:

import numpy as np
import random
from matplotlib import pyplot as plt

img_arr = np.empty([20, 220])
for j in range(0, 10):
    img_row = np.empty([20, 20])
    for i in range(0, 10):
        img = X[random.randint(1, 5000)].reshape(20, 20)
        img_row = np.concatenate((img_row, img), axis=1)
    img_arr = np.concatenate((img_arr, img_row), axis=0)

plt.figure(figsize=(12,8))
plt.imshow(img_arr[20:, 20: ].T, interpolation="nearest", cmap='gray')

# Show random selection of 100 images using imshow
plt.show()


# # 3. Vectorizing the Cost function

# In[3]:

from scipy.special import expit

def h(theta, vec):
    return expit(np.dot(vec, theta.T))

def cost_function(theta, X, Y):
    first_term = -(np.log(h(theta, X)) * Y.T)
    second_term = -(np.log(1.0-h(theta, X)) * (1.0-Y.T))
    return float(np.sum(first_term + second_term)/Y.shape[0])


# # 4. Vectorized Gradient Descent

# In[34]:

theta = np.zeros((1, 400))
def gradient(theta, X, Y):
    Y = np.array([Y]).T
    gradient = np.array(np.dot(X.T, (h(theta, X) - Y)).T/Y.shape[0])
    return gradient.reshape(1, 400)


# # 5. Regularization

# ## 5.1 Regularizing the Cost function

# In[35]:

def regularized_cost_function(theta, X, Y):
    regular_cost = cost_function(theta, X, Y)
    lambda_ = 100
    cost = regular_cost + (lambda_/(2*Y.shape[0]))*np.sum(theta[:, 1:] * theta[:, 1:])
    return cost


# ## 5.2 Regularizing the Gradient

# In[39]:

def regularized_gradient(theta, X, Y):
    regular_gradient = gradient(theta, X, Y)
    lambda_ = 100
    regularization_vec = np.copy(theta)
    regularization_vec = regularization_vec * (lambda_/Y.shape[0])
    regularization_vec[0, 0] = 0
    return regularization_vec

