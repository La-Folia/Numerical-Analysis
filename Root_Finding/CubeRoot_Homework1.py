#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
plt.rcParams.update({'font.size': 15})


# In[5]:


from scipy import optimize
def f(x):
    return (x**3 - 2)

def df(x):
    return 3*x**2

sol = optimize.root_scalar(f, x0 = 1, fprime = df,
                         method = 'newton')

sol.root, sol.iterations

