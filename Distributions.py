
# coding: utf-8

# # Data Distributions

# ## Uniform Distribution

# In[1]:

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

values = np.random.uniform(-10.0, 10.0, 100000)
plt.hist(values, 50)
plt.show()


# ## Normal / Gaussian

# Visualize the probability density function:

# In[2]:

from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.001)
plt.plot(x, norm.pdf(x))


# Generate some random numbers with a normal distribution. "mu" is the desired mean, "sigma" is the standard deviation:

# In[3]:

import numpy as np
import matplotlib.pyplot as plt

mu = 5.0
sigma = 2.0
values = np.random.normal(mu, sigma, 10000)
plt.hist(values, 50)
plt.show()


# ## Exponential PDF / "Power Law"

# In[4]:

from scipy.stats import expon
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.001)
plt.plot(x, expon.pdf(x))


# ## Binomial Probability Mass Function

# In[5]:

from scipy.stats import binom
import matplotlib.pyplot as plt

n, p = 10, 0.5
x = np.arange(0, 10, 0.001)
plt.plot(x, binom.pmf(x, n, p))


# ## Poisson Probability Mass Function

# Example: My website gets on average 500 visits per day. What's the odds of getting 550?

# In[6]:

from scipy.stats import poisson
import matplotlib.pyplot as plt

mu = 500
x = np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu))

