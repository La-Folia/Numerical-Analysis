#!/usr/bin/env python
# coding: utf-8

# # SciPy

# ### SciPy는 Python의 numpy확장을 기반으로 구축된 수학적 알고리즘 함수 모음

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.12.43.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.12.43.png)

# ## Bessel Function

# In[11]:


from matplotlib import pyplot as plt
import numpy as np
from scipy import special

x = np. linspace(0, 20, 100)
y = [special.jn(n, x) for n in (0, 1, 2)]

plt.figure()

plt.plot(x, y[0], '-r', label = "$j_0(x)$")
plt.plot(x, y[1], '--g', label = "$j_1(x)$")
plt.plot(x, y[2], '-.b', label = "$j_2(x)$")

plt.xlabel("$x$")
plt.legend()
plt.grid()
plt.show()


# In[65]:


n = 1
alpha = special.jn_zeros(n, 10) #정수의 베셀 함수를 계산.


x = np.linspace(0, np.ceil(np.max(alpha)), 200)
#x = np. linspace(0, 40, 200) 와 같다.
y = special.jv(n, x)

#x = np. linspace(0, 20, 100)
#y = [special.jn(n, x) for n in (0, 1, 2)]

plt.figure()

plt.plot(x, y, "-")
plt.plot(alpha, 0 * alpha, 'ok', label = f"solution of $J_{n}(x) = 0$")
plt.xlabel(f"$x$")
plt.ylabel(f"$J_{n}(x)$")

plt.grid()
plt.legend()
plt.show()


# In[61]:


np.ceil(np.max(alpha))


# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.54.14.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.54.14.png)

# In[64]:


alpha = special.jn_zeros(n, 10)
alpha


# In[41]:


n = 1
alpha = special.jn_zeros(n, 10)
x = np.linspace(0, np.ceil(np.max(alpha)), 200)
y = special.jv(n, x)

plt.figure()

plt.plot(x, y, "-")
plt.plot(alpha, 0.001 * alpha, 'ok', label = f"solution of $J_{n}(x) = 0$")
plt.xlabel(f"$x$")
plt.ylabel(f"$J_{n}(x)$")
plt.grid()
plt.legend()

plt.show()

