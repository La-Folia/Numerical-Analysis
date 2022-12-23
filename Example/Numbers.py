#!/usr/bin/env python
# coding: utf-8

# Numerical Analysis and Practice

# Integers

# In[16]:


a = 8 + 12
print(f"a = {a}, type of a = {type(a)}")

b = 100 / 4
print(f"b = {b}, type of b = {type(b)}")

c = 100// 4
print(f"c = {c}, type of c = {type(c)}")


# Floats

# In[18]:


float(1)


# In[20]:


1.0


# In[22]:


2**0.5


# In[24]:


1.41423423


# In[29]:


5e-10 * 3


# Complex Numbers j = root(-1)

# In[32]:


complex(1,2)


# In[34]:


1 + 2j


# In[36]:


1j * (1 + 2j)


# Exercises

# Taylor Approximation

# In[40]:


1 + 0.5 + 0.5 **2 / 2 + 0.5 **3 / (3 * 2) + 0.5 ** 4 / (4 * 3 * 2) + 0.5 ** 5 / (5 * 4 * 3 * 2 * 1)


# In[41]:


1 + 1/ (2+ 1/(2 + 1/(2+ 1 / 2)))

