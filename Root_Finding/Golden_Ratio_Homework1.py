#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from scipy import optimize


# In[4]:


def f(x):
    return (x**2 - x - 1)

sol = optimize.root_scalar(f, bracket=[1, 2], method='bisect')

