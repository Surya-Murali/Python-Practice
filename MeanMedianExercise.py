
# coding: utf-8

# # Exercise: Mean & Median Customer Spend

# Here's some code that will generate some random e-commerce data; just an array of total amount spent per transaction. Select the code block, and hit "play" to execute it:

# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(100.0, 20.0, 10000)

plt.hist(incomes, 50)
plt.show()


# Now, find the mean and median of this data. In the code block below, write your code, and see if your result makes sense:

# In[5]:

incomes.mean()
np.median(incomes)


# This is pretty much the world's easiest assignment, but we're just trying to get your hands on iPython and writing code with numpy to get you comfortable with it.
# 
# Try playing with the code above to generate different distributions of data, or add outliers to it to see their effect.

# In[ ]:



