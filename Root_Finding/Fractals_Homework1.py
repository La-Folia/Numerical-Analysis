#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
#%config InlineBackend.figure_format='retina'


# In[43]:


from scipy import optimize
def f(x):
    return (x**3 - 1)  

def fprime(x):
    return 3*x**2
    

def found_real_root(zinit: complex) -> int:
    sol = optimize.root_scalar(f, 
                                x0 = 1,
                                fprime = fprime,
                                method = 'newton')
    x, y = sol.root.real, sol.root.imag
    print(x, y)
    if abs(y) > 1e-10 or abs(x - 1) > 1e-10:
        return 0

    return 1

xs = np.linspace(-2, 2, 700)
ys = xs
mask = [[found_real_root(complex(x, y)) for x in xs] for y in ys]

plt.figure(figsize=(11, 11))
plt.imshow(mask, interpolation='none', origin='lower', cmap='Greys',
            extent=np.hstack((xs[[0, -1]], ys[[0, -1]])))
plt.xlabel('$\\Re(z)$')
plt.ylabel('$\\Im(z)$')
plt.show()


# In[26]:


root = optimize.root_scalar(f, 
                                x0 = 1j,
                                fprime = fprime,
                                method = 'newton')


# In[28]:


root.iterations

