
# coding: utf-8

# # Polynomial Regression

# In[2]:

get_ipython().magic('matplotlib inline')
from pylab import *
import numpy as np

np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

scatter(pageSpeeds, purchaseAmount)


# numpy has a handy polyfit function we can use, to let us construct an nth-degree polynomial model of our data that minimizes squared error. Let's try it with a 4th degree polynomial:

# In[59]:

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

p4 = np.poly1d(np.polyfit(x, y, 4))
p4
np.polyfit(x, y, 4)


# We'll visualize our original scatter plot, together with a plot of our predicted values using the polynomial for page speed times ranging from 0-7 seconds:

# In[60]:

import matplotlib.pyplot as plt

xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()


# Looks pretty good! Let's measure the r-squared error:

# In[82]:

from sklearn.metrics import r2_score

r2 = r2_score(y, p4(x))

print(r2)

