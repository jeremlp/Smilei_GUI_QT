# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:24:41 2025

@author: Jeremy
"""
import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
import matplotlib.ticker as ticker

plt.close("all")

def Ftheta_V2_O3(r,theta,z):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3
# @njit
def LxEpolar_V2_O3(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O3(r,theta,z)

def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def f_prime(r,z):
    C_lp = np.sqrt(1/math.factorial(abs(l1)))
    return C_lp/w(z)**3 * exp(-(r/w(z))**2) * (r/w(z))**(abs(l1)-1) * (-2*r**2+w(z)**2*abs(l1))
def f_squared_prime(r,z):
    return 2*w0**2/(w(z)**2*r) * f(r,z)**2*(abs(l1)-2*(r/w0)**2+ 4*(z**2/w0**4))

l0=2*pi

w0=2.5*l0
Tp=12*l0
a0 = 0.1
l1=1

x0=5*l0



dx_interp = 0.02*l0
y_range = np.arange(-2*w0,2*w0,dx_interp)
Y, Z = np.meshgrid(y_range,y_range, indexing='ij')
THETA = np.arctan2(Z,Y)
R = np.sqrt(Y**2+Z**2)

extent = [y_range[0]/l0,y_range[-1]/l0,y_range[0]/l0,y_range[-1]/l0]



fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
Lx_NR = LxEpolar_V2_O3(R,THETA,0,w0,a0,3/8*Tp)
im1= ax1.imshow(Lx_NR, extent=extent, cmap="RdYlBu")
fig.colorbar(im1,ax=ax1,pad=0.01)

Lx_NR = LxEpolar_V2_O3(R,THETA,5*l0,w0,a0,3/8*Tp)
im2 = ax2.imshow(Lx_NR, extent=extent, cmap="RdYlBu")
fig.colorbar(im2,ax=ax2,pad=0.01)

ax1.set_xlabel("$y_0/\lambda$")
ax1.set_ylabel("$z_0/\lambda$")
ax1.set_title("$L_x^{NR}(x,y)$ at $x=0$")
ax2.set_title("$L_x^{NR}(x,y)$ at $x=5\lambda$")
fig.tight_layout()

#==========================================

r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R,THETA = np.meshgrid(r_range,theta_range)

plt.figure(figsize=(10,4))
Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,x0))**2+ 1/4*(a0*f(r_range,x0))**4)
plt.plot(r_range/l0,Lx_max_model,"C0-")
plt.plot(r_range/l0,-Lx_max_model,"C0-", label="$x=0$")
plt.fill_between(r_range/l0,Lx_max_model,-Lx_max_model,alpha=0.25,color="C0")

Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,5*l0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,x0))**2+ 1/4*(a0*f(r_range,x0))**4)
plt.plot(r_range/l0,Lx_max_model,"C1-")
plt.plot(r_range/l0,-Lx_max_model,"C1-", label="$x=5\lambda$")
plt.fill_between(r_range/l0,Lx_max_model,-Lx_max_model,alpha=0.25,color="C1")
plt.grid()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.title("Radial distribution $L_x^{NR}(r_0)$")

plt.legend()