            #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:11:44 2022

@author: kang0
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
#%config InlineBackend.figure_format='retina'

from scipy import special

x = np.linspace(0, 30, 200)
y = special.jn(0, x)

plt.figure()
plt.plot(x, y, '-r', label="$J_0(x)$")
plt.xlabel("$x$")
plt.ylabel("$J_0(x)$")
plt.grid()
#plt.show()
plt.savefig("12.pdf")
