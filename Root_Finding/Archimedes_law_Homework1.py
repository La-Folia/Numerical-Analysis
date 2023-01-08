#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from scipy import optimize


# In[9]:


def arch(h) :
    r = 5
    p = 1
    o = 0.6
    return (1/3 * np.pi * (3 * r * h**2 - h**3) * p) - (4/3 * np.pi * r**3 * o)

def fprime(h):
    r = 5
    p = 1
    o = 0.6
    return 1/3 * np.pi * (6 * r * h - 3 * h**2) * p

sol = optimize.root_scalar(arch, x0 = 2, fprime = fprime, 
                          method='newton')
#초기값 x0에 따라 값이 두개가 나온다.
sol.root


# In[121]:


h = np.arange(-20, 20)
r = 5
p = 1
o = 0.6
y = (1/3 * np.pi * (3 * r * h**2 - h**3) * p) - (4/3 * np.pi * r**3 * o)

plt.plot(h, y)
plt.grid(color='0.5')
plt.show()


# In[90]:


a =(1/3 * np.pi * p) /  (4/3 * np.pi * r**3 * o)
print(a)


# In[69]:


#Test
h = 13.305409517702934
print(1/3 * np.pi * (3 * r * h**2 - h**3) * p)
print(4/3 * np.pi * r**3 * o)

