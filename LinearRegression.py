
# coding: utf-8

# # Linear Regression

# Let's fabricate some data that shows a roughly linear relationship between page speed and amount purchased:

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
from pylab import *

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

scatter(pageSpeeds, purchaseAmount)


# As we only have two features, we can keep it simple and just use scipy.state.linregress:

# In[16]:

from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
stats.linregress(pageSpeeds, purchaseAmount)


# Not surprisngly, our R-squared value shows a really good fit:

# In[19]:

r_value ** 2


# In[ ]:

Let's use the slope and intercept we got from the regression to plot predicted values vs. observed:


# In[34]:

import matplotlib.pyplot as plt

def predict(x):
    return slope * x + intercept

fitLine = predict(pageSpeeds)
plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()
#fitLine
pageSpeeds


# In[ ]:



