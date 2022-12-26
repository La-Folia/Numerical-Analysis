#!/usr/bin/env python
# coding: utf-8

# # Built- in Functions
# ### 재사용을 막아주는 함수에 대한 여러가지 분석
# 
# #### Built- in Functions
# 
# ##### abs(x) : x의 절댓값을 돌려줌, 복소수일 경우 계수
# ##### int(x) : 정수를 돌려준다
# ##### sorted(sequence) : list를 정렬해줌
# ##### reversed(sequence) : list를 반대로 정렬(내림차순)
# ##### enumerate(sequence) : 시퀀스에서 생성된 열거 개체를 반환합니다. -->열거해서 나열하는 듯
# ##### zip(a, b) : 시퀀스 a 및 b에서 항목을 집계하는 반복 가능한 개체를 반환합니다.
# ##### map(fun, sequence) : 주어진 함수를 각 항목에 순차적으로 적용한 후 결과의 반복자를 반환합니다.
# ##### filter(pred, sequence) : 특정 조건을 충족하는 모든 시퀀스 요소에 대해 반복기를 반환합니다.

# In[1]:


x = -594.939
abs(x)


# In[2]:


z = 3 - 4j
abs(z)


# In[3]:


pi = 3.14159
int(pi)


# In[4]:


c = -1.4948
int(c)


# In[5]:


random = [8, 27, 3, 7, 6, 14, 28, 19]
sorted_random = sorted(random)
print(random)
print(sorted_random)


# In[6]:


reversed_range = reversed(range(10))
reversed_range


# In[7]:


tuple(reversed_range)


# In[8]:


#squares = [[i ** 1, i **2, i ** 3] for i in range(10)]
#list(enumerate(squares))
#잘 활용하면 무궁무진하게 효율적이게 사용할 수 있는 함수..


# #### enumerate 개인 연습

# In[23]:


oii = [[i,i+1] for i in range(1000000)]
list(enumerate(oii))


# In[10]:


integers = range(10)
list(zip(integers, reversed(integers)))


# In[11]:


def my_enum(seq):
    return list(zip(range(len(seq)), seq))

my_enum(squares)


# In[12]:


list(map(lambda x: x ** 2, range(10)))


# In[13]:


list(filter(lambda x: x % 2 == 0, range(10)))


# In[14]:


list(i for i in range(10) if i % 2 == 0)


# ### Lambdas

# ### laambda parametes: expression
# ### def <lambda> (parameters):
#     return expression

# In[26]:


my_enum2 = lambda seq: list(zip(range(len(seq)), seq))
my_enum2(squares)


# ## User-defined Functions

# In[29]:


def average(x):
    return sum(x) / len(x)

average(range(10))


# In[30]:


def is_even(i):
    return i % 2 == 0

list(filter(is_even, range(20)))

