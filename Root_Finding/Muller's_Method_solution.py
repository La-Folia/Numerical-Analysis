#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
#%config InlineBackend.figure_format='retina'


# # Muller's Method
# 
# [Muller's method](https://en.wikipedia.org/wiki/Muller%27s_method) generalizes the secant method but uses *quadratic* interpolation
# 
# $$
# Q(t) = At^2 + Bt + C
# $$
# 
# among *three* points intead of linear interpolation between two.
# Solving for the zeros of the quadratic $Q(t) = 0$ allows the method to find complex pairs of roots:
# 
# $$
# t = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A}
# =
# -\frac{2C}{B \pm \sqrt{B^2 - 4AC}}
# $$
# 
# As you will see below, we would like to choose the root closer to zero, which means that we need to choose the sign that results in the smallest numerator in the first form or the largest denominator in the second form.
# In numerical calculation, the second form is perferable to avoid round-off errors due to subtraction of nearby equal numbers.

# In[ ]:





# ## Muller's Formula
# 
# <center><figure>
#   <img src="https://media.geeksforgeeks.org/wp-content/uploads/Muller-Method.png" width="400">
#   <figcaption>Image from https://www.geeksforgeeks.org/program-muller-method/</figcaption>
# </figure></center>
# 
# A function $y = f(x)$ goes through the three points, $(x_0, y_0)$, $(x_1, y_1)$, and $(x_2, y_2)$.
# We would like to find a root of $f(x) = 0$ closer to $x_2$.
# By making change of variable $t = x - x_2$, the problem becomes finding a root $t$ closer to $0$.
# Demanding $Q(0) = y_2$, $Q(t_1) = y_1$, and $Q(t_0) = y_0$ with $t_1 = x_1 - x_2$ and $t_0 = x_0 - x_2$, a parabola that goes through the three points is constructed with the coefficients
# 
# $$
# \begin{aligned}
# A &= \frac{1}{t_0 - t_1} \left(\frac{y_0 - y_2}{t_0} - \frac{y_1 - y_2}{t_1}\right) \\
# B &= -\frac{1}{t_0 - t_1} \left(t_1\frac{y_0 - y_2}{t_0} - t_0\frac{y_1 - y_2}{t_1}\right) \\
# C &= y_2 \\
# \end{aligned}
# $$
# 
# The next approximation $x_3$ is produced by the following formulas,
# 
# $$
# x_3 = x_2 - \frac{2C}{B \pm \sqrt{B^2 - 4AC}}
# $$
# 
# where the sign in the denominator is chosen such that $\left|B \pm \sqrt{B^2 - 4AC}\right|$ is maximized.

# ## Algorithm
# 
# 1. Starting from the three initial values ${x_0, x_1, x_2}$, find the next approximation $x_3$ according to Muller's formula:
# 
#     $$
#     x_3 = x_2 - \frac{2C_2}{B_2 \pm \sqrt{B_2^2 - 4A_2C_2}}
#     $$
#     
# 2. We then use this new value $x_3$ and repeat the process, using ${x_3, x_2, x_1}$ as the initial values, solving for $x_4$, $x_5$, etc.:
# 
#     $$
#     \begin{aligned}
#     x_4 &= x_3 - \frac{2C_3}{B_3 \pm \sqrt{B_3^2 - 4A_3C_3}}, \\
#     x_5 &= x_4 - \frac{2C_4}{B_4 \pm \sqrt{B_4^2 - 4A_4C_4}}, \\
#     &\ \vdots \\
#     x_n &= x_{n - 1} - \frac{2C_{n - 1}}{B_{n - 1} \pm \sqrt{B_{n - 1}^2 - 4A_{n - 1}C_{n - 1}}}, \\
#     \end{aligned}
#     $$
#     
# 3. The iteration stops when we reach a sufficiently high level of precision (a sufficiently small difference between $x_n$ and $x_{n - 1}$).
# 
# *Note* that $x_n$ can be complex.

# ## Implementation

# In[ ]:


import cmath

def muller_while(f, x0x1x2, predicate):
    """Return the root calculated using Muller's method.
    
    :param f:
        A function f(x).
    :param x0x1x2:
        Three initial guesses.
    :param predicate:
        A callable that accepts three arguments:
        A predicate function which takes three arguments
            - i : the iteration count
            - xy : a pair of the midpoint and the function value in the current iteration
            - dx : the change of the x value
        and returns boolean:
            - If True, the search continues.
            - If False, the search terminates.
    """
    
    x0, x1, x2 = map(complex, x0x1x2)
    f0, f1, f2 = f(x0), f(x1), f(x2)
    i, x3, f3 = 0, float("nan"), float("nan")
    
    def muller_root():
        nonlocal i, x3, f3
        #
        t0   , t1    =  x0 - x2    ,  x1 - x2
        term0, term1 = (f0 - f2)/t0, (f1 - f2)/t1
        denom = t0 - t1
        #
        A = (   term0 -    term1) / denom
        B = (t0*term1 - t1*term0) / denom
        C = f2
        #
        sqrt_discriminant = cmath.sqrt(B**2 - 4*A*C)
        dx = -2*C / max(B + sqrt_discriminant, B - sqrt_discriminant, key=abs)
        x3 = x2 + dx
        f3 = f(x3)
        i += 1
        return i, (x3, f3), dx
    
    while predicate(*muller_root()):
        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f3
        
    return x3


# Let's calculate the supergolden ratio.

# In[ ]:


supergolden = muller_while(lambda x: x ** 3 - x ** 2 - 1, (1, 2, 3), 
                           lambda i, xy, dx: abs(xy[1]) > 1e-10)
supergolden


# *Notice* the result can be complex.

# ## Exercises

# ### CubeRoot
# 
# Approximate $\sqrt[3]{2}$ by solving
# 
# $$
# x^3 - 2 = 0
# $$

# In[ ]:


cuberoot2_approx = muller_while(lambda x: x*x*x - 2, (1, 1.5, 2),
                                lambda i, xy, dx: abs(dx) > 1e-10)
(cuberoot2_approx, abs(2**(1/3) - cuberoot2_approx))


# ### Complex Logarithm
# 
# Consider a function
# 
# $$
# f(z) = i\log z + \frac{\pi}{2}
# $$
# 
# We know that a solution of $f(z) = 0$ is $z = i$.
# Find the solution of $f(z) = 0$ numerically.

# In[1]:


z0z1z2 = (1 + .1j, 1 + .5j, 1 + .9j)

z_approx = muller_while(lambda z: 1j * cmath.log(z) + cmath.pi/2, z0z1z2,
                        lambda i, zf, dz: abs(zf[1]) > 1e-10)

print(f"real(z) = {z_approx.real}, imag(z) = {z_approx.imag}")


# ### Two-stream Instability
# 
# The [two-stream instability](https://en.wikipedia.org/wiki/Two-stream_instability) is a very common instability in plasma physics.
# It can be induced by an energetic particle stream injected in a plasma, or setting a current along the plasma so different species (ions and electrons) can have different drift velocities.
# The energy from the particles can lead to plasma wave excitation.
# 
# Consider a cold, uniform, and unmagnetized plasma, where ions are stationary and the electrons have velocity $v_0$.
# The dispersion relation is
# 
# $$
# 1 = \omega_p^2 \left[\frac{m_e/m_i}{\omega^2} + \frac{1}{(\omega - k v_0)^2}\right]
# $$
# 
# where $\omega$ is the wave frequency, $k$ the wave number, $m_e/m_i$ the ratio of electron-to-ion mass, and $\omega_p$ is the electron plasma frequency.
# 
# Let $f_k(\omega)$
# 
# $$
# f_k(\omega) = \omega_p^2 \left[\frac{m_e/m_i}{\omega^2} + \frac{1}{(\omega - k v_0)^2}\right] - 1
# $$
# 
# For $k \rightarrow \infty$, we find
# 
# $$
# f_k(\omega) \approx \omega_p^2 \frac{m_e/m_i}{\omega^2} - 1
# $$
# 
# Therefore, the solution of $f_k(\omega) = 0$ for large $k$ should approach
# 
# $$
# \omega(k \rightarrow \infty) = \sqrt{\frac{m_e}{m_i}} \omega_p
# $$
# 
# Assuming, $v_0 = \omega_p = 43$ and $m_e/m_i = 1/43^2$,
# we search for roots of $f_k(\omega) = 0$ starting from a large value of $k$ and moving to small values:

# In[ ]:


# define f_k(w)
def dispersion_relation(k, w):
    v0, wp, me_mi = 43, 43, 1 / 43**2
    return 1 - wp**2 * (me_mi/w**2 + 1/(w - k*v0)**2)

# define a function that solves the relation for a given k
def single_step(k, winit):
    max_w = max(winit, key=abs)
    return muller_while(lambda w: dispersion_relation(k, w), winit,
                        lambda i, xy, dw: abs(dw) > 1e-7 * abs(max_w))

# define the k space
ks = np.linspace(2, .03, 1000)

# first three guesses of omega
# we know for large k the frequency should approach wp/√(m_e/m_i) = 1
ws = [1 + .1j, 1, 1. - .1j]

# walk over the k values and find the solutions
for k in ks:
    ws.append(single_step(k, ws[-3:]))

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


# We find that the wave frequency becomes complex-valued in the region of small $k$,
# indicating wave growth/damping in that region.
