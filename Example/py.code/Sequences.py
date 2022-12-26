#!/usr/bin/env python
# coding: utf-8

# In[4]:


squares = [1, 4, 9, 16, 25]
squares


# In[8]:


points = [[0,0], [0,1], [1, 1], [0,1]]
points


# In[11]:


heterogeneous_list = [[0,0], [], [1], [0,"1"]]
heterogeneous_list


# In[13]:


primes = [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 29]
primes[0]


# In[15]:


primes[-1]


# In[18]:


primes[0] = -1
primes


# In[22]:


pairs = [[0,1], [2, 3], [4, 5], [6, 7]]
pairs[2][1]


# In[24]:


fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
fibonacci[4:7]


# In[26]:


fibonacci[6:]


# In[27]:


fibonacci[:-2]


# In[29]:


fibonacci[0:11:3]


# In[31]:


one = [1]
two = [2, 2]
three = [3, 3, 3]
numbers = one + two + three
numbers


# In[33]:


(one + two) * 3


# In[35]:


squares = [1, 4, 9, 16, 25]
squares.append(36)
squares


# In[37]:


today = (2019, 7, 11)
today


# In[38]:


type((1))


# In[40]:


(1,) + (2,3)


# In[43]:


today[0] = 2022


# Range Objects

# In[45]:


digits_range = range(0, 10)
digits_range


# In[46]:


tuple(digits_range)


# In[48]:


even_list = list(range(0, 10, 2))
even_list


# Unpacking a Sequence

# In[50]:


today  = (2019, 7, 11)
year, month, day = today
print(f"Date: {year}/{month}/{day}")


# In[53]:


(one, two, three) = range(1, 4)
print(f"range(1, 4 ) => {one}, {two}, and {three}")


# In[56]:


args = 0, 10, 2
list(range(*args))


# In[58]:


print("range(1, 4) => {}, {}, and {}".format(*range(1, 4)))


# In[60]:


args = range(0, 10, 2)
(*args,)


# List Comprehensions

# [expression for item in iterable]

# In[62]:


squares = []
for x in range(10):
    squares.append(x**2)
    
squares


# In[64]:


squares = [y ** 2 for y in range(10)]
squares


# In[67]:


print(x)
#print(y)


# In[72]:


even_list = [x for x in range(10) if x % 2 ==0]
even_list


# In[81]:


matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]

transposed = [
    [row[i] for row in matrix]
    for i in range(len(matrix[0]))
]
transposed


# In[87]:


for i in range(len(matrix[0])): 
    for row in matrix:
        print(row[i])
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:


transposed = [list(entry) for entry in zip(*matrix)]
transposed


# In[80]:


list(map(list, zip(*matrix)))


# https://www.daleseo.com/python-zip/ zip function?

# Generator Expressions

# (expression for item in iterable)

# In[92]:


sum(i ** 2 for i in range(1, 11))


# Built- in Functions for Sequences

# In[91]:


len([1, 2, 3])


# In[97]:


random = [3, -5, 7, 8, -1]
print(f"sum = {sum(random)}")
print(f"max = {max(random)}")
print(f"min = {min(random)}")


# In[99]:


sorted(random)


# In[100]:


one_to_hundred = range(1, 101)
sum(one_to_hundred)


# Exercies

# In[102]:


N = 1000
rhs = N * (N + 1) // 2
lhs = sum(i + 1 for i in range(N))
rhs == lhs


# Riemann Zeta Function

# In[104]:


N = 1000
approx_zeta_2 = sum(1 / n ** 2 for n in range(N+1)[1:])
approx_zeta_2


# Maclaurin Series

# In[106]:


N = 5000
x = 1
approx_actran_1 = sum(
    (pow(-1, n) * pow(x, 2 * n + 1)) / (2 * n + 1)
    for n in range(N + 1)
)
approx_actran_1

