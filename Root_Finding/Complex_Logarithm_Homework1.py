#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from scipy import optimize
import cmath


# In[48]:


z_approx = optimize.root_scalar(lambda z: 1j * cmath.log(z) + cmath.pi/2,  
                          x0= 1 + .1j, x1 = 1 + .5j, fprime=lambda z: 1j / z,
                                method='secant')


z_approx

