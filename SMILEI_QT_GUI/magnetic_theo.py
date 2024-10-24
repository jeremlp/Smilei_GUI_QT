# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:14:12 2024

@author: jerem
"""
import numpy as np
from numpy import exp, cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import math

plt.close("all")
l0 = 2*pi

a0 = 10
w0 = 3.5*l0
Tp=30*l0
eps,l= 0,1
zR = 0.5*w0**2
C_lp = np.sqrt(2/math.factorial(abs(l)))

"""
FROM Berezhiani (1997)
"""

def f(r,z):
    return C_lp * (r/w(z))**abs(l)*exp(-1.0*(r/w(z))**2)
def f_prime(r,z):
    return C_lp/w(z)**3 * exp(-(r/w(z))**2) * (r/w(z))**(abs(l)-1) * (-2*r**2+w(z)**2*abs(l))
def w(z):
    return w0*sqrt(1+(z/zR)**2)
def a(z):
    return a0*w0/w(z)/sqrt(1+abs(eps))

ne = 0.03
def N(r):
    res = 1 - 1/(ne*w0**2)*(gamma**2-1)/gamma * (2 - r**2/w0**2*(gamma**2+1)/gamma**2)
    res[res <0] = 1e-5
    return res

r = np.arange(0,4*w0,0.1)
gamma = 1+a0*f(r,0)

def p_theta(r):
    res = 2*(gamma**2-1)/gamma * np.gradient(np.log(N(r)/gamma),r)
    # res[np.isnan(res)] = 0
    return res

plt.plot(r/l0,N(r),label='N')
Lx = r*p_theta(r)
plt.plot(r/l0,Lx/np.nanmax(np.abs(Lx)), label='Lx')


Bx = 1/r*np.gradient(Lx,r)
plt.plot(r/l0,Bx/np.nanmax(np.abs(Bx[np.abs(Bx) != np.inf])), label='Bx')
plt.legend()
plt.grid()


