#!/usr/bin/env python
# coding: utf-8

# # NumPy

# ### Numpy Arrays

# In[1]:


import numpy as np


# In[2]:


a = np.array([1, 2, 3, 4, 5])
print(a)


# In[3]:


print([1, 2, 3, 4, 5])


# In[4]:


a


# In[5]:


type(a)


# In[6]:


M = np.array([
    [1, 2, 3],
    [4, 5, 6]
    
])
print(M)


# In[7]:


print(np.array(10))


# In[8]:


a = np.fromiter(reversed(range(10)), int)
print(a)


# In[9]:


np.fromiter(reversed(range(10)), int)


# In[10]:


np.arange(10, 30, 5)


# In[11]:


np.arange(0, 2, 0.3)


# In[12]:


np.zeros(4, int)


# In[13]:


print(np.zeros((3,4)))


# In[14]:


print(np.ones((2,3)))


# In[15]:


print(np.eye(10, dtype = int))


# ### Array Datatype

# In[16]:


A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(A)


# In[17]:


A.dtype


# In[18]:


u = np.linspace(0, 1, 5)
print(u)


# In[19]:


u.dtype


# In[20]:


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
A.ndim


# In[21]:


A.shape


# In[22]:


r = np.array(range(10))
print(r)


# In[23]:


r.ndim


# In[24]:


r.shape


# In[25]:


r.size


# ## Slicing and Indexing

# In[26]:


v = np.arange(10)
v[3]


# In[27]:


B = np.array([
    [6, 5, 3, 1, 1],
    [1, 0, 4, 0, 1],
    [5, 9, 2, 2, 9]
])
B[1, 2]


# In[28]:


B[1][2]


# In[29]:


B[1:3, 2:5]


# In[30]:


B[-1, -2]


# In[31]:


B[2, :]


# In[32]:


B[2]


# In[33]:


B[:, 3]


# In[34]:


subB = B[1:3, 2:5]
print(subB)


# In[35]:


subB.ndim


# In[36]:


subB.shape


# In[37]:


subB.size


# In[38]:


colB = B[:, 2]
print(colB)


# In[39]:


colB.ndim


# In[40]:


colB.shape


# In[41]:


colB.size


# ## Staking

# In[42]:


x = np.array([1, 1, 1])
y = np.array([2, 2, 2])
z = np.array([3, 3, 3])
vstacked = np.vstack([x, y, z])
print(vstacked)


# In[43]:


hstacked = np.hstack([x, y, z])
print(hstacked)


# In[44]:


A = 1 * np.ones((2, 2))
B = 2 * np.ones((2, 2))
C = 3 * np.ones((2, 2))
D = 4 * np.ones((2, 2))


# In[45]:


print(np.vstack([
    np.hstack([A, B]),
    np.hstack([C,D])
]))


# In[46]:


print(np.vstack([np.hstack([A, B]), np.hstack([C, D])]))


# ## Numpy 배열 복사 대 보기

# In[47]:


arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)


# In[48]:


arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)


# In[49]:


x[0] = 31
print(arr)
print(x)


# In[50]:


arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)


# ## Operations and Functions

# ### Array Operations

# In[51]:


v = np.array([1, 2, 3])
w = np.array([1, 0, -1])

v + w


# In[52]:


v - w


# In[53]:


v * w


# In[54]:


w / v


# In[55]:


w // v


# In[56]:


print( w ** v)
print( v ** 2)


# In[57]:


a = np.array([1,1])
a ** -1


# In[ ]:


A = np.array([
    [3,1],
    [2, -1]
])
B = np.array([
    [2, -2],
    [5, 1]
])


# In[ ]:


A + B


# In[ ]:


A - B 


# In[ ]:


A / B


# In[ ]:


A * B


# In[ ]:


A ** 2


# In[ ]:


A @ B #행렬의 곱을 의미함****


# ## 행렬 거듭제곱은 함수에 의해 수행된다. Numpy.linalg.matrix_power을 이용함.

# In[ ]:


from numpy.linalg import matrix_power as mpow
mpow(A, 3)


# In[ ]:


A @ A @ A


# # BroadCasting!!!!

# In[58]:


x = np.linspace(0, 1, 5)
y = x **2 + 1
print(y)


# In[59]:


u = np.array([1, 2, 3, 4])
A = np.array([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
])
result = A + u
print(result)
print(A)


# In[60]:


A.transpose() + u


# In[61]:


A.shape = (3, 1, 4)
print(A)
print(A + u)


# In[62]:


A.shape = (1, 3, 4)
A + u


# In[64]:


u.shape = 4, 1
A + u


# ## 배열 함수

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.56.34.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.56.34.png)

# In[66]:


arr = np.array([8, -2, 4, 7, -3])
print(arr)


# In[68]:


arr.mean()


# In[71]:


print(f"max element = {arr.max()}, its index = {arr.argmax()}")


# In[73]:


M = np.array([
    [2, 4, 2],
    [2, 1, 1],
    [3, 2, 0],
    [0, 6,2 ]
])
print(M)


# In[75]:


M.sum()


# In[77]:


M.sum(axis = 0)


# In[79]:


M.sum(axis = 1)


# ## Math Function

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.43.13.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.43.13.png)

# In[97]:


x = np.arange(0, 1.01, 0.25)
np.sin(2 * np.pi * x)


# In[173]:


b = map(np.sin, 2 * np.pi * x) 
# map(a, b)에서 리스트 [b1, b2, b3] 의 x값을 
# function인 f(y)에 대입하여 
# [f(b1), f(b2), f(b3)]로 만들어줌.
list(b)


# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.43.54.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.43.54.png)

# In[110]:


x = 10 ** np.array(range(5))
np.log10(x)


# In[112]:


np.pi


# In[114]:


np.e


# ## 난수 생성기

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.49.25.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.49.25.png)

# ### 균일 분포에서의 난수

# In[116]:


np.random.rand()


# In[118]:


np.random.rand(3)


# In[134]:


np.random.rand(2, 4)


# ### 표준 정규 분포에서의 임의의 표본

# In[123]:


np.random.randn()


# In[125]:


np.random.randn(3)


# In[135]:


np.random.randn(2, 3)


# ### 다양한 간격에서 균일하게 샘플링된 무작위 정수

# In[129]:


np.random.randint(-10, 10)


# In[131]:


np.random.randint(0, 2, (4, 8))


# In[136]:


np.random.randint(-9, 10, (5, 2))


# # 운동

# ## 무차별 대입 최적화

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.14.28.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.14.28.png)

# In[157]:


N = 1000
x = np.linspace(0, 2 * np.pi, N)
y = x * np.sin(x) + np.cos(4 * x)

i_minmax = [f(y) for f in (np.argmin, np.argmax)]
x_minmax = x[i_minmax]
y_minmax = y[i_minmax]

print(f'Absolute maximum value is y = {y_minmax[1]} at x = {x_minmax[1]}')
print(f'Absolute minimum value is y = {y_minmax[0]} at x = {x_minmax[0]}')


# In[159]:


i_minmax = [f(y) for f in (np.argmin, np.argmax)]
y[i_minmax]


# In[198]:


b = map(np.sin, 2 * np.pi * x) 


# # 리만 합계

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.58.37.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.58.37.png)

# In[200]:


def midpoint_rule(f, xs):
    assert (len(xs) > 1)
    
    xi = (xs[:-1] + xs[1:]) / 2
    dx = -(xs[:-1] - xs[1:])
    
    return np.sum(f(xi) * dx)


# In[202]:


midpoint_rule(lambda x: np.exp(-(x * x)), 
             np.linspace(0, 100, 100000))


# # 무한 제품

# ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.58.53.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.58.53.png)

# In[205]:


def cos_product(x, N):
    n = np.arange(N) + 1
    second_term = ((2 * x) / (np.pi * (2 * n - 1))) ** 2
    return np.prod(1 - second_term)


# In[207]:


cos_product(0, 10)


# In[210]:


cos_product(np.pi, 10000)


# In[211]:


(cos_product(np.pi / 4, 10000000), 1 / 2 ** 0.5)

