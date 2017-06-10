
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

def sigmoid(vec):
    return 1/(1 + np.exp(-vec))

def h(theta, vec):
    return sigmoid(np.dot(vec, theta.T))

def cost_function(theta, X, Y):
    return float((np.sum(-np.dot(Y, np.log(h(theta, X)))) - np.dot((1-Y), np.log(1-h(theta, X))))/
                 Y.shape[0])


# # 4. Vectorized Gradient Descent

# In[7]:

theta = np.zeros((1, 400))
def gradient(theta, X, Y):
    Y = np.array([Y]).T
    gradient = np.array(np.dot(X.T, (h(theta, X) - Y)).T/Y.shape[0])
    return gradient

