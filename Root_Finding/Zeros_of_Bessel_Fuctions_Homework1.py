#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from scipy import optimize


# In[6]:


from scipy import special

x = np.linspace(0, 30, 200)
y = special.jn(0, x)

plt.figure()
plt.plot(x, y, '-r', label="$J_0(x)$")
plt.xlabel("$x$")
plt.ylabel("$J_0(x)$")
plt.grid()
plt.savefig("Bessel_Functions.pdf")


# In[8]:


intervals = (0, 5, 7, 10, 13, 16, 20, 23, 25, 28)

x_zeros = []
def f(x):
    return special.jn(0, x)

for ab in zip(intervals[:-1], intervals[1:]):
    sol = optimize.root_scalar(f, bracket=ab, method='bisect')
    print(ab)
    x_zeros.append(sol.root)


plt.figure()
plt.plot(x, y, '-r')
plt.plot(x_zeros, np.zeros_like(x_zeros), 'ok',
         label="$J_0=0$ points")
plt.xlabel("$x$")
plt.ylabel("$J_0(x)$")
plt.grid()
plt.legend()
plt.savefig("Zeros_of_Bessel_Fuctions_Homework1.pdf")

