#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
#%config InlineBackend.figure_format='retina'


# # Bisection Method
# 
# The simplest root finding algorithm is the [bisection method](https://en.wikipedia.org/wiki/Bisection_method).
# The algorithm applies to any continuous function $f(x)$.
# 
# <center><figure>
#   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Bisection_method.svg/250px-Bisection_method.svg.png">
#   <figcaption>From Wikipedia (https://en.wikipedia.org/wiki/Bisection_method)</figcaption>
# </figure></center>
# 
# Suppose the value of $f(x)$ changes sign from $a$ to $b$.
# A solution of the equation $f(x) = 0$ in the interval $[a, b]$ is guaranteed by the [intermediate value theorem](https://en.wikipedia.org/wiki/Intermediate_value_theorem), provided that $f(x)$ is continuous on $[a, b]$.
# In other words, the function changes sign over the interval $[a, b]$ and therefore must equal $0$ at some point in the interval.
# 
# The idea of the bisection method is simple:
# 
# - Divide the interval in two, a solution must exist within one subinterval
# - Select the subinterval where the sign of $f(x)$ changes
# - Repeat until a solution is found (within the desired accuracy)

# ## Algorithm
# 
# The bisection method procedure is:
# 
# 1. Choose a starting interval $[a_0, b_0]$ within which we know a root is (i.e., $f(a_0) f(b_0) < 0$).
# 2. Compute $f(m_0)$ where $m_0 = (a_0 + b_0)/2$ is the midpoint.
# 3. Determine the next subinterval $[a_1, b_1]$:
# 
#     - If $f(a_0) f(m_0) < 0$, then let $[a_1, b_1]$ with $a_1 = a_0$ and $b_1 = m_0$ be the next interval.
#     - If $f(m_0) f(b_0) < 0$, then let $[a_1, b_1]$ with $a_1 = m_0$ and $b_1 = b_0$ be the next interval.
#     
# 4. Repeat (2) and (3) until the interval $[a_n, b_n]$ reaches some predetermined criterion.
# 5. Return the midpoint value $m_n = (a_n + b_n)/2$.
# It is this value that is our approximate solution $f(m_n) \approx 0$.
# 
# The bisection method is one that *cannot* fail:
# 
# - If the interval happens to contain more than one root, bisection will find one of them.
# - If the interval contains no roots and merely straddles a singularity, it will converge on the singularity.

# ## Absolute Error
# 
# It is clear from the algorithm that after each iteration, the bounds containing the root decrease by a factor of two.
# That is, if after $n$ iterations, the root is known to be within an interval of size $\epsilon_n$, 
# then after the next iteration it will be bracketed within an interval of size
# 
# $$
# \epsilon_{n + 1} = \epsilon_n/2
# $$
# 
# Thus, we know in advance the number of iterations required to achieve a given tolerance in the solution:
# 
# $$
# n = \log_2 \frac{\epsilon_0}{\epsilon}
# $$
# 
# where $\epsilon_0$ is the size of the initially bracketing interval (i.e., $|b - a|$) and $\epsilon$ is the desired ending tolerance.
# 
# Let $x_{\rm true}$ be the exact solution and $x_n$ the approximate one after $n$ iterations.
# Then, the absolute error after $n$ iterations is
# 
# $$
# |x_{\rm true} - x_n| \le \epsilon = \frac{\epsilon_0}{2^{n + 1}}
# $$
# 
# (The extra factor $1/2$ comes from the fact that we are returning the midpoint of the subinterval after $n$ iterations.)

# ## Implementation
# 
# Write a function called `bisection_by` which takes four input parameters `f`, `a`, `b` and `N` and returns the approximation of a solution of $f(x) = 0$ given by $n$ iterations of the bisection method.
# If $f(a_n) f(b_n) \ge 0$ at any point in the iteration, then print `"Bisection method fails."` and return `None`.

# In[3]:


def bisection_by(f, a, b, n):
    """Approximate solution of f(x) = 0 on interval [a, b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x) = 0.
    a, b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    n : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_n : number
        The midpoint of the nth interval computed by the bisection method. The
        initial interval [a_0, b_0] is given by [a, b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.
    """

    a_n = a
    f_of_a_n = f(a_n)
    b_n = b
    f_of_b_n = f(b_n)

    # validity check
    if f_of_a_n * f_of_b_n >= 0:
        print("Bisection method fails.")
        return None

    # iterations
    m_n = 0.5 * (a_n + b_n)
    f_of_m_n = f(m_n)

    for _ in range(n):
        if f_of_m_n == 0:
            print("Found exact solution.")
            return m_n

        elif f_of_a_n * f_of_m_n < 0:
            b_n = m_n
            f_of_b_n = f_of_m_n

        elif f_of_b_n * f_of_m_n < 0:
            a_n = m_n
            f_of_a_n = f_of_m_n

        else:
            print("Bisection method fails.")
            return None

        m_n = 0.5 * (a_n + b_n)
        f_of_m_n = f(m_n)

    return m_n


# It is often useful to monitor iterations from outside by supplying a monitor function.
# Furthermore, you may want to decide when to terminate the iteration by observing the intermediate values.
# 
# Let's define a function called `bisection_while`:

# In[4]:


def bisection_while(f, xinit, predicate):
    """Approximate solution of f(x) = 0 on interval xinit = [a, b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x) = 0.
    xinit : a pair of numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    predicate : callable
        A function that takes three arguments:
            - i : the iteration count
            - xy : a pair of the midpoint and the function value in the current iteration
            - dx : the change of the x value
        and should return boolean:
            - If True, the search continues.
            - If False, the search terminates.

    Returns
    -------
    x_n : number
        The midpoint of the nth interval computed by the bisection method. The
        initial interval [a_0, b_0] is given by [a, b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.
    """

    a_n, b_n = xinit
    f_1st = f(a_n)

    # check if initial interval is valid
    if f(a_n) * f(b_n) >= 0:
        print("Bisection method fails.")
        return None

    # iterations
    i = 1
    x_mid = 0.5 * (a_n + b_n)
    f_mid = f(x_mid)
    while predicate(i, (x_mid, f_mid), 0.5 * abs(a_n - b_n)):
        if f_1st * f_mid > 0:
            a_n = x_mid
            f_1st = f_mid
        else:
            b_n = x_mid

        i = i + 1
        x_mid = 0.5 * (a_n + b_n)
        f_mid = f(x_mid)

    return x_mid


# ## Exercises

# ### CubeRoot
# 
# Approximate $\sqrt[3]{2}$ by solving
# 
# $$
# x^3 - 2 = 0
# $$

# In[5]:


cuberoot2_approx = bisection_while(lambda x: x*x*x - 2, (1, 2),
                                   lambda i, xy, dx: abs(dx) > 1e-10)
(cuberoot2_approx, abs(2**(1/3) - cuberoot2_approx))


# ### Golden Ratio
# 
# The [golden ratio](https://en.wikipedia.org/wiki/Golden_ratio) is given by
# 
# $$
# \phi = \frac{1 + \sqrt 5}{2} \approx 1.6180339887498948482
# $$
# 
# which is one solution of $f(x) \equiv x^2 - x - 1 = 0$.
# 
# Use the bisection implementation with $N = 25$ iterations on $[1, 2]$ to approximate $\phi$.

# In[6]:


approx_phi = bisection_by(lambda x: x*(x - 1) - 1, 1, 2, 25)
approx_phi


# The absolute error is guaranteed to be less than $(2 - 1)/2^{26}$:

# In[7]:


error_bound = 2 ** (-26)
abs((1 + 5 ** 0.5) / 2 - approx_phi) < error_bound


# Let's do the same thing with `bisection_while`:

# In[8]:


approx_phi == bisection_while(lambda x: x*(x - 1) - 1, (1, 2),
                              lambda i, xy, dx: i <= 25)


# Let's visualize how the bisection method searches for the solution using `bisection_while`:

# In[9]:


ix_pairs  = []
iy_pairs  = []
idx_pairs = []


def intercept(i, xy, dx):
    ix_pairs.append([i, xy[0]])
    iy_pairs.append([i, abs(xy[1])])
    idx_pairs.append([i, abs(dx)])
    return i <= 11


bisection_while(lambda x: x*(x - 1) - 1, (1, 2), intercept)


# In[10]:


plt.figure(figsize=(11, 4))

plt.subplot(1, 3, 1)
plt.plot(*zip(*ix_pairs), 'o:k', label="bisection")
plt.plot([1, len(ix_pairs)], (1 + 5 ** 0.5) / 2 * np.ones((2,)), '--r',
         label="solution")
plt.title("$x$ approximation")
plt.xlabel("iteration count")
plt.ylim([1, 2])
plt.legend()

plt.subplot(1, 3, 2)
plt.semilogy(*zip(*iy_pairs), 'o:k')
plt.title("$|f(x_n)|$")
plt.xlabel("iteration count")
plt.grid()

plt.subplot(1, 3, 3)
plt.semilogy(*zip(*idx_pairs), 'o:k', label="$\Delta x_n$")
plt.semilogy(*zip(*[(idx[0], 2/2**idx[0]) for idx in idx_pairs]),
             'r--', label="$\Delta x \propto 2^{-n}$")
plt.xlabel("iteration count")
plt.title("|$\Delta x$|")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# ### Zeros of Bessel Functions
# 
# Let's find the zeros of Bessel functions, $J_0(x) = 0$.
# 
# First, we need to know the intervals bracketing the solutions.

# In[11]:


from scipy import special

x = np.linspace(0, 30, 200)
y = special.jn(0, x)

plt.figure()
plt.plot(x, y, '-r', label="$J_0(x)$")
plt.xlabel("$x$")
plt.ylabel("$J_0(x)$")
plt.grid()
plt.show()


# From the figure above, the first zero is bracketed by $[0, 5]$, the second by $[5, 7]$, and so on.
# Using this information we can apply our implementation of the bisection method.

# In[12]:


intervals = (0, 5, 7, 10, 13, 16, 20, 23, 25, 28)
x_zeros = [
    bisection_while(lambda x: special.jn(0, x), ab,
                    lambda i, xy, dx: abs(xy[1]) >= 1e-10)
    for ab in zip(intervals[:-1], intervals[1:])
]

plt.figure()
plt.plot(x, y, '-r')
plt.plot(x_zeros, np.zeros_like(x_zeros), 'ok',
         label="$J_0=0$ points")
plt.xlabel("$x$")
plt.ylabel("$J_0(x)$")
plt.grid()
plt.legend()
plt.show()


# Let's compare how close our solutions are to the exact values.

# In[13]:


abs_err = [abs(x - y) for x, y in zip(x_zeros, special.jn_zeros(0, len(x_zeros)))]
abs_err

