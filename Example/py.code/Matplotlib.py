#!/usr/bin/env python
# coding: utf-8

# # Basic Plotting

# ## Procedure

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
x = [-5, -2, 0, 1, 3]
y = [2, -1, 1, -4 ,3]
plt.plot(x, y)
plt.show()


# In[8]:


x = np.linspace(-2, 2, 100)
y = x ** 2
plt.plot(x, y)
plt.show()


# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.00.41.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.00.41.png)

# In[26]:


x = np.linspace(-2, 2, 41)
y = np.exp(-x ** 2) * np.cos(2 * np.pi * x)
plt.plot(x, y,
        alpha = 0.4, label = 'Decaying Cosine',
        color = 'red', linestyle = 'dashed',
        linewidth = 1, marker = 'o',
        markersize = 3, markerfacecolor = 'blue',
        markeredgecolor = 'blue'
        )

plt.ylim([-2, 2])
plt.legend()
#plt.savefig("14.pdf") --> pdf로 저장하는 법 


# ## 형식 문자열

# fmt = '[marker][line][color]'

# In[28]:


x = np.linspace(-5, 5, 41)
y = 1/ (1 + x ** 2)
plt.plot(x, y,
        color = 'black',
        linestyle = 'dashed',
        marker = 's'
        )
plt.show()


# In[30]:


plt.plot(x, y, 'ks--')
plt.show()


# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.15.54.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.15.54.png)

# In[53]:


def factorial(n):
    assert(n > 0)
    return np.prod(np.arange(1, n + 1))

x = np.linspace(-1, 1, 200)
y = np.exp(x)

fN_minus_1 = 1
for N in range(1, 5):
    fN = fN_minus_1 + x ** N / factorial(N)
    
    plt.subplot(2, 2, N)
    plt.plot(x, y, 'k-', label = "$f_{N}$")
    plt.plot(x, fN, 'r--', label = f"$f_{N}$")
    plt.title(f"N = {N}")
    plt.legend() 

    plt.xlim([f(x) * 1.1 for f in (np.min, np.max)])
    plt.ylim([f(y) * 1.1 for f in (np.min, np.max)])
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    fN_minus_1 = fN #1이 끝나고 다시 2가 되어야 하는데, 1에서 곱해준 값으로 취해버리면 안되니, 값을 초기화 시켜주기 위해, 다시 1로 만듬.
    
plt.tight_layout()
plt.show()


# In[60]:


N = 2000
points = np.random.rand(2, N)
sizes = np.random.randint(20, 120, (N,))

colors = np.random.rand(N, 4)

plt.figure(figsize = (12, 5))
plt.scatter(*points, c = colors, s = sizes)
plt.axis('off')
plt.show()


# In[66]:


samples = np.random.randn(10000)
plt.hist(samples,
        bins = 20, density = True,
        alpha = 0.5, color = (0.3,0.8,0.1))
plt.title('Random Samples = Normal Distribution')
plt.ylabel('PDF')

x = np.linspace(-4, 4, 100)
y = 1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * x ** 2)
plt.plot(x, y, 'b-', alpha = 0.8)

plt.show()


# In[87]:


x = np.linspace(-1, 1, 50) * np.pi * 2
y  = np.cos(x)
plt.plot(x, y, 'b', label = 'cos(x)')

y2 = 1 - x ** 2 / 2
plt.plot(x, y2, 'r-.', label = 'Degree 2')

y4 = 1- x**2 / 2 + x ** 4 / 24
plt.plot(x, y4, 'g:', label = 'Degree 4')

plt.legend(loc = 'upper center')
plt.grid(True, linestyle = ':')

plt.xlim([f(x) * 1 for f in (np.min, np.max)])
plt.ylim([f(y) * 3 for f in (np.min, np.max)])

plt.title('Taylor Polynomials of cos(x) at x = 0')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


# In[88]:


np.linspace(-1, 1, 50) * np.pi * 2


# In[98]:


t = np.linspace(0, 2 * np.pi, 100)

x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos (2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

plt.plot(x, y, c = (1, 0.2, 0.5), lw = 5)

plt.title('Heart!')
plt.axis('equal')
plt.axis('off')

plt.show()


# In[124]:


plt.subplot(2, 2, 1)
x = np.linspace(-9, 9, 200)
y = np.sqrt(np.abs(x))
plt.plot(x, y , 'b')
plt.title("function 1")

plt.subplot(2, 2, 2)
x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x) + np.sin(2 * x)
plt. plot(x, y, 'b')
plt.title("function 2")

plt.subplot(2, 2, 3)
x = np.linspace(-5, 5, 200)
y = np.arctan(x)
plt. plot(x, y, 'b')
plt.title("function 3")

plt.subplot(2, 2, 4)
x = np.linspace(-2, 3, 200)
y = np.array([x + a for a in [2, 1, -1, -2, -3]]).prod(0) 
#2, 1, -1...일때를 각각 행렬로 만들고 그걸 요소들을 곱하면, 
# 위의 식과 같은 결과를 내보내겠지.
plt. plot(x, y, 'b')
plt.title("function 4")


# In[126]:


np.array([x + a for a in [2, 1, -1, -2, -3]]).prod(0)


# In[134]:


plt.figure(figsize = (5, 10))

t = np.linspace(0, 2 * np.pi, 200)

plt.subplot(3, 1, 1)
plt.plot(np.sin(t), np.sin(t) * np.cos(t), 'b')
plt.title("Figure 8")

plt.subplot(3, 1, 2)
plt.plot(np.sin(t) + 2 * np.sin(2 * t), np.cos(t) -2 * np.cos(2 * t), 'b')
plt.title("Trefoil knot")

t = np.linspace(0, 12 * np.pi, 2000)
plt.subplot(3, 1, 3)
plt.plot(np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5),
         np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5),
                    'b')
plt.title("Trefoil knot")

plt.tight_layout()
plt.show()


# In[ ]:




