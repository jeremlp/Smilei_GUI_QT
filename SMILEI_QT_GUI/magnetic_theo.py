# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:14:12 2024

@author: jerem
"""
import numpy as np
from numpy import exp, cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import math
import os
import sys
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/')
l0=2*np.pi

plt.close("all")
l0 = 2*pi

a0 = 10
w0 = 6*l0
Tp=30*l0
eps,l= 0,0
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
gamma = np.sqrt(1+(a0*f(r,0))**2)

def p_theta(r):
    res = 2*(gamma**2-1)/gamma * np.gradient(np.log(N(r)/gamma),r)
    # res[np.isnan(res)] = 0
    return res

plt.plot(r/l0,N(r),label='N')
Lx = r*p_theta(r)
plt.plot(r/l0,Lx/np.nanmax(np.abs(Lx)), label='Lx')


Bx = 1/r*np.gradient(Lx,r)
plt.plot(r/l0,Bx/np.nanmax(np.abs(Bx[np.abs(Bx) != np.inf])), label='Bx')
plt.grid()
plt.xlabel("$r/\lambda$")
plt.title(f"Normalized quantities, from Berezhiani (1997)\na0={a0}, w0={w0/l0:.1f}$\lambda$, $\epsilon$={eps}, l={l}")

weight_diag = S.ParticleBinning("weight_av")
weight = np.mean(np.array(weight_diag.getData())[-1,:,:]/0.03,axis=-1)
weight_av = np.mean(weight, axis=0)
y_range_w = (weight_diag.getAxis("y")-S.namelist.Ltrans/2)/l0

Bx_av_long_diag = S.Field("Bx_av","Bx_m")
Bx = np.array(Bx_av_long_diag.getData())
y_range = (Bx_av_long_diag.getAxis("y")-S.namelist.Ltrans/2)/l0

mean_Bx =np.mean(Bx[-1],axis=0)

Bx1 = mean_Bx[y_range<=0]
Bx2 = mean_Bx[y_range>=0]

w1 = weight_av[y_range_w<=1e-5]
w2 = weight_av[y_range_w>=0]

# plt.figure()
# plt.plot(np.abs(y_range_w[y_range_w<=1e-5]),w1)
# plt.plot(np.abs(y_range_w[y_range_w>=0]),w2)

Bx_mean = np.mean([Bx1[::-1],Bx2],axis=0)
w_mean = np.mean([w1[::-1],w2],axis=0)
# plt.plot(np.abs(y_range_w[y_range_w>=0]),w_mean,"k--")

# plt.plot(np.abs(y_range[y_range>=0]),Bx_mean/np.max(np.abs(Bx_mean)),"k-.",label="OAM Bx")
# plt.plot(np.abs(y_range_w[y_range_w>=0]),w_mean,"k--",label="OAM N")
plt.legend()

# plt.grid()

