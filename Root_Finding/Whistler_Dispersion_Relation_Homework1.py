#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
plt.rcParams.update({'font.size': 15})


# In[202]:


def f(x): 
    c, wp, wc = 5, 5, 1
    f  = c**2 * k**2 - x**2 + wp**2/(1 - wc/x)
    return f

def df(w) :
    c, wp, wc = 5, 5, 1
    df = -2*w - wp**2 * wc/(w - wc)**2
    return df

ks = np.linspace(0.01, 10, 40)

for ab in zip(ks[:-1], ks[1:]):
    print(ab)
    sol = optimize.root_scalar(f, bracket=ab, method='bisect') 
    ws.append(sol.root)
    

w_exact = np.linspace(1e-10, 0.99, 400)
k_exact = 1/5 * np.sqrt(w_exact**2 - 5**2/(1 - 1/w_exact))

plt.figure()
plt.plot(k_exact, w_exact, "-k", label = "$\\omega$ exact")
plt.plot(ks, ws, ".:r", linewidth = 1, label = "$\\omega$ approx")
plt.xlabel("$k$")
plt.ylabel("$\\omega$")
plt.legend()
plt.show()


# bisect와 branq 등의 방법은 f(a) 와 f(b)가 부호가 달라야 한다. 즉, 진동하는 함수 값에서 해를 구할 때만 사용 가능하다
# 
# >루트를 괄호로 묶는 간격. f(x, *args)는 두 끝점에 다른 기호가 있어야 합니다.
# 

# In[251]:


def f(k): 
    c, wp, wc = 5, 5, 1
    f  = w*2 #c**2 * k**2 - w**2 + wp**2/(1 - wc/w)
    return f

def df(k) :
    c, wp, wc = 5, 5, 1
    df = -2*w - wp**2 * wc/(w - wc)**2
    return df


ks = np.linspace(0.1, 10, 40)
ws = [ks[0]]

for k in ks:
    sol = optimize.root_scalar(lambda w, k: f(w, k), fprime = lambda w, k: df(w, k), x0 = 1, 
                        method = 'newton')
    ws.append(sol.root)
    
'''
ws = [ws.append(sol.root) for k in np.linspace(0.01, 10, 40)]
'''


ws = ws[1:]

w_exact = np.linspace(1e-10, 0.99, 400)
k_exact = 1/5 * np.sqrt(w_exact**2 - 5**2/(1 - 1/w_exact))

plt.figure()
plt.plot(k_exact, w_exact, "-k", label = "$\\omega$ exact")
plt.plot(ks, ws, ".:r", linewidth = 1, label = "$\\omega$ approx")
plt.xlabel("$k$")
plt.ylabel("$\\omega$")
plt.legend()
plt.show()


# In[236]:


ks


# In[1]:


ks = np.linspace(0.1, 10, 40)
ws = [ks[0]]

def f(k): 
    c, wp, wc = 5, 5, 1
    f  = w*2 #c**2 * k**2 - w**2 + wp**2/(1 - wc/w)
    return f


def df(k) :
    c, wp, wc = 5, 5, 1
    df = -2*w - wp**2 * wc/(w - wc)**2
    return df

for k in ks:
    sol = optimize.root_scalar(f, fprime = df, x0 = 1, 
                        method = 'newton')
    print(sol)

