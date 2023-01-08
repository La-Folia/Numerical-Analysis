#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
#%config InlineBackend.figure_format='retina'


# # Newton's Method
# 
# Perhaps the most celebrated of all one-dimensional root-finding routines is [*Newton's method*](https://en.wikipedia.org/wiki/Newton%27s_method), also called the *Newton-Raphson method*.
# This method is distinguished from the previous methods by the fact that it requires the evaluation of both the function $f(x)$ *and* the derivative $f'(x)$, at arbitrary points $x$.
# 
# <center><figure>
#     <img src=https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Newton_iteration.svg/300px-Newton_iteration.svg.png>
#     <figcaption>From Wikipedia (https://en.wikipedia.org/wiki/Newton%27s_method)</figcaption>
# </figure></center>
# 
# The Newton-Raphson formula consists geometrically of extending the tangent line at a current point $x_n$ until it crosses zero, then setting the next guess $x_{n + 1}$ to the abscissa of that zero crossing:
# 
# $$
# x_{n + 1} = x_n - \frac{f(x_n)}{f'(x_n)}
# $$
# 
# Algebraically, the method derives from the familiar Taylor series expansion of a function in the neighborhood of a point
# 
# $$
# f(x + \delta) \approx
# f(x) + f'(x)\delta + \frac{f''(x)}{2} \delta^2 + \cdots
# $$
# 
# For small enough values of $\delta$, and for well-behaved functions, the terms beyond linear are unimportant, hence $f(x + \delta) = 0$ implies
# 
# $$
# \delta = -\frac{f(x)}{f'(x)}
# $$
# 
# The Newton-Rapshon method is not restricted to one dimension; the method readily generalizes to multiple dimensions.

# ## Pros and Cons
# 
# When it converges, Newton's method usually converges very quickly and this is its main advantage.
# However, Newton's method is not guaranteed to converge and this is obviously a big disadvantage especially compared to the bisection method which is guaranteed to converge to a solution (provided they start with an interval containing a root).
# For example, if a trial guess is near a local extremum so that the first derivative $f'(x)$ nearly vanishes, then Newton's method sends the next guess far off from the actual root.
# 
# Newton's method also requires computing values of the derivative of the function in question.
# This is potentially a disadvantage if the derivative is difficult to compute.

# ## Implementation

# In[1]:


def newton_while(f_and_df, xinit, predicate):
    """Return the root calculated using Newton's method.
    
    :param f_and_df:
        A function that returns a pair of numbers containing f(x) and f'(x)
    :param xinit:
        A trial guess
    :param predicate:
        A predicate function which takes three arguments
            - i : the iteration count
            - xy : a pair of the midpoint and the function value in the current iteration
            - dx : the change of the x value
        and returns boolean:
            - If True, the search continues.
            - If False, the search terminates.
    """
    
    x = xinit
    i = 0

    def netwon_root():
        nonlocal i, x
        f, df = f_and_df(x)
        dx = -f / df
        x_old = x
        x += dx
        i += 1
        return i, (x_old, f), dx
    
    while predicate(*netwon_root()):
        pass
        
    return x


# Let's calculate the supergolden ratio.

# In[ ]:


def predicate(i, xy, dx):
    print("i = {:1d}, x = {:.12f}, y = {:+.12e}, dx = {:+.12e}".format(i, *xy, dx))
    return abs(dx) > 1e-10 and i < 20

newton_while(lambda x: (x**3 - x**2 - 1, 3*x**2 - 2*x), 1, predicate)


# It found the solution after 7 iterations.
# Test how many iterations it would take for the bisection method starting with the interval $[1, 2]$.

# ### Divergent Example
# 
# As alerted earlier, Newton's method diverges in certain cases,
# for example, if the tangent line at the root is vertical as in $f(x) = x^{1/3}$.

# In[1]:


iterations = -1
def predicate(i, xy, dx):
    global iterations
    iterations = i
    print("\ti = {}, xy = {}, dx = {}".format(i, xy, dx))
    return abs(dx) > 1e-10 and i < 20

solution = newton_while(lambda x: (x**(1/3), (1/3) * x**(-2/3)), 0.1, predicate)

if iterations >= 20:
    print(f"\n!! Exceeded maximum iterations (={20}) before convergence !!")

print(solution)


# ## Exercises

# ### CubeRoot
# 
# Approximate $\sqrt[3]{2}$ by solving
# 
# $$
# x^3 - 2 = 0
# $$

# In[ ]:


cuberoot2_approx = newton_while(lambda x: (x**3 - 2, 3*x*x), 1,
                                lambda i, xy, dx: abs(dx) > 1e-10)
cuberoot2_approx, abs(2**(1/3) - cuberoot2_approx)


# ### Whistler Dispersion Relation
# 
# Calculate the whistler dispersion relation we did previously, now using Newton's method.
# 
# We have defined
# 
# $$
# f_k(\omega) = 
# c^2 k^2 - \omega^2 + \frac{\omega_p^2}{1 - \omega_c/\omega}
# $$
# 
# Its derivative is
# 
# $$
# f_k'(\omega) = 
# -2\omega - \omega_p^2\frac{\omega_c}{(\omega - \omega_c)^2}
# $$

# In[ ]:


# define f_k(w) and f_k'(w)
def f_and_df(k, w):
    c, wp, wc = 5, 5, 1
    f  = c**2 * k**2 - w**2 + wp**2/(1 - wc/w)
    df = -2*w - wp**2 * wc/(w - wc)**2
    return f, df

# define a function that solves the relation for a given k
def single_step(k, winit):
    return newton_while(lambda w: f_and_df(k, w), winit,
                        lambda i, wf, dw: abs(dw) > 1e-7 * abs(max(wf[0], winit, key=abs)))

# define the k space
ks = np.linspace(0.01, 10, 40)

# initial trial guess of omega
ws = [ks[0]]

# walk over k values and find solutions
for k in ks:
    ws.append(single_step(k, ws[-1]))

# remove the first guess
ws = ws[1:]

w_exact = np.linspace(1e-10, 0.99, 400)
k_exact = 1/5 * np.sqrt(w_exact**2 - 5**2/(1 - 1/w_exact))

plt.figure()
plt.plot(k_exact, w_exact, "-k", label="$\\omega$ exact")
plt.plot(ks, ws, ".:r", linewidth=1, label="$\\omega$ approx")
plt.xlabel("$k$")
plt.ylabel("$\\omega$")
plt.legend()
plt.show()


# ### Fractals
# 
# In mathematics, [fractal](https://en.wikipedia.org/wiki/Fractal) is a term used to describe geometric shapes containing detailed structure at arbitrarily small scales.
# 
# Newton's method can be applied to complex-valued functions and solutions.
# The solution of the equation
# 
# $$
# z^3 - 1 = 0
# $$
# 
# has a single real root $z = 1$.
# 
# Choose an array of trial guesses from a complex domain spanning $[-2, 2]$ in real and imaginary axes.
# Paint the region where the trial guesses led to the real root in black; otherwise in white.

# In[ ]:


def found_real_root(zinit: complex) -> int:
    root = newton_while(lambda z: (z**3 - 1, 3*z**2), zinit,
                        lambda i, xy, dx: abs(dx) > 1e-10 and i < 20)
    x, y = root.real, root.imag
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


# Let's zoom in the region $\Re(z) \in [-0.5, 0.5]$ and $\Im(z) \in [-0.5, 0.5]$.

# In[ ]:


def found_real_root(zinit: complex) -> int:
    root = newton_while(lambda z: (z**3 - 1, 3*z**2), zinit,
                        lambda i, xy, dx: abs(dx) > 1e-10 and i < 20)
    x, y = root.real, root.imag
    if abs(y) > 1e-10 or abs(x - 1) > 1e-10:
        return 0
    return 1

xs = np.linspace(-.5, .5, 700)
ys = xs
mask = [[found_real_root(complex(x, y)) for x in xs] for y in ys]

plt.figure(figsize=(11, 11))
plt.imshow(mask, interpolation='none', origin='lower', cmap='Greys',
            extent=np.hstack((xs[[0, -1]], ys[[0, -1]])))
plt.xlabel('$\\Re(z)$')
plt.ylabel('$\\Im(z)$')
plt.show()

