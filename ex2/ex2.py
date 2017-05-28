
# coding: utf-8

# # Exercise 2: Logistic Regression

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# ## 1. Plotting the data

# In[13]:

data = np.loadtxt('data/ex2data1.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)
data = data.T
X = data[:,0:2]
Y = data[:,2]
pos = data[Y == 1]
neg = data[Y == 0]

plt.figure(figsize=(12,8))
points_pos, = plt.plot(pos[:, 0], pos[:, 1], 'g+', markersize=10, label="Admitted")
points_neg, = plt.plot(neg[:, 0], neg[:, 1], 'ro', markersize=7, label="Not admitted")
plt.legend(handles=[points_pos, points_neg])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()


# ## 2. Sigmoid Function

# In[32]:

def sigmoid(vec):
    return 1/(1 + np.exp(-vec))

