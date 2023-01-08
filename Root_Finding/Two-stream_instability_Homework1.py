#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from scipy import optimize
import cmath


# In[8]:


# define f_k(w)
'''
def dispersion_relation(k, w):
    v0, wp, me_mi = 43, 43, 1 / 43**2
    return 1 - wp**2 * (me_mi/w**2 + 1/(w - k*v0)**2)

# define a function that solves the relation for a given k
def single_step(k, winit):
    max_w = max(winit, key=abs)
    return muller_while(lambda w: dispersion_relation(k, w), winit,
                        lambda i, xy, dw: abs(dw) > 1e-7 * abs(max_w))
'''

# define the k space
ks = np.linspace(2, .03, 1000)

# first three guesses of omega
# we know for large k the frequency should approach wp/√(m_e/m_i) = 1
ws = [1 + .1j, 1, 1. - .1j]

def dispersion_relation(k, w):
    v0, wp, me_mi = 43, 43, 1 / 43**2
    return 1 - wp**2 * (me_mi/w**2 + 1/(w - k*v0)**2)

# walk over the k values and find the solutions
for k in ks:
    sol = optimize.root_scalar(dispersion_relation(k, w), bracket = [2,.03] ,method='brentq')

# remove the first three guesses
ws = ws[3:]

plt.figure()
plt.plot(ks, [w.real for w in ws], "-b", linewidth=1, label="$\\Re(\\omega)$")
plt.plot(ks, [w.imag for w in ws], "-r", linewidth=1, label="$\\Im(\\omega)$")
plt.xlabel("$k$")
plt.ylabel("$\\omega$")
plt.legend()
plt.grid()
plt.show()


# In[ ]:




