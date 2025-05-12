# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:05:41 2025

@author: Jeremy
"""

# %% INIT

import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = f"{os.environ['SMILEI_SRC']}"
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
import time
from tqdm import tqdm
from smilei_utils import SET_CONSTANTS, averageAM, min_max,min_max_percentile
from smilei_utils import dr_func, dx_func, theta_func, dtheta_func

plt.close("all")

l0 = 2*pi


w0 = 2.5*l0
a0=1
Tp=12*l0

x0 = 5*l0
SET_CONSTANTS(w0, a0, Tp)

# %% FUNCTIONS

def Torque_V2_O5(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    numerator = ( 2 * a0**2 * r**2 * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (2 * z * np.cos(2 * theta) * ( 4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)) +
            np.sin(2 * theta) * (  4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * ( 8 * (w0**4 + 4 * w0**2 + 20) * z**2 + (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4) -
                (w0**4 + 4 * z**2) * ( 4 * (w0**2 - 2) * z**2 + (w0**2 + 4) * (w0**2 + 6) * w0**2  ))))
    denominator = (w0**4 + 4 * z**2)**5
    return numerator / denominator
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

def Ftheta_V2_O3(r,theta,z):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3
# @njit
def Ftheta_V2_O5(r,theta,z):
    numerator = (
        2 * a0**2 * r * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (2 * z * np.cos(2 * theta) * ( 4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)) +
            np.sin(2 * theta) * (  4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * ( 8 * (w0**4 + 4 * w0**2 + 20) * z**2 + (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4) -
                (w0**4 + 4 * z**2) * ( 4 * (w0**2 - 2) * z**2 + (w0**2 + 4) * (w0**2 + 6) * w0**2  ))))
    denominator = (w0**4 + 4 * z**2)**5
    return numerator / denominator


def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)

zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)
def g_cst(tau):
    return  1
def g_gauss(tau):
    return  exp( -((tau-1.25*Tp)/(Tp*3/8/c_gauss))**2) 
def g_sin2(tau):
    return  sin(pi*tau/Tp)**2*(tau>0)*(tau<Tp)
g = g_cst
def getE(r,theta,z,t):
    tau = t-z
    
    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l1)+1)*arctan2(z,zR)
    a_ = a0*w0/w_/sqrt(1+abs(eps))
    f_ = C_lp * (r/w_)**abs(l1)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l1)-1) * (-2*r**2+w_**2*abs(l1))
    
    Ex = a_*f_*g(tau)*cos(l1*theta + phi_)
    Ey = -eps*a_*f_*g(tau)*sin(l1*theta + phi_)
    Ez = -a_*(f_prime_*g(tau)*cos(theta))*sin(l1*theta + phi_) +\
        a_*f_*g(tau)*(l1/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l1*theta+phi_)
    return Ex,Ey,Ez

# %% MAIN CODE
t_range = np.arange(0, 4*pi,0.05)
y_range = np.arange(-1.5*w0,1.5*w0,0.1)
Y,Z,T = np.meshgrid(y_range, y_range, t_range, indexing="ij")
R = np.sqrt(Y**2+Z**2)
THETA = np.arctan2(Z,Y)

Ey,Ez,Ex = getE(R,THETA,x0,T)

Int_Ey = np.mean(Ey**2,axis=-1)
Int_Ex = np.mean(Ex**2,axis=-1)

extent = [y_range[0]/l0,y_range[-1]/l0, y_range[0]/l0,y_range[-1]/l0]

cmap = "jet"

# %% PLOT

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
im1 = ax1.imshow(Int_Ey,cmap=cmap, extent=extent)
fig.colorbar(im1, ax=ax1,pad=0.01)
ax1.set_xlabel("y")
ax1.set_xlabel("z")
ax1.set_title("|$E_y^2$|")

im2 = ax2.imshow(Int_Ex,cmap=cmap, extent=extent)
fig.colorbar(im2,ax=ax2,pad=0.01)
ax2.set_xlabel("y")
ax2.set_ylabel("z")
ax2.set_title("|$E_x^2$|")


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
im1 = ax1.imshow(np.log10(Int_Ey),cmap=cmap, vmin=-1.5, extent=extent)
fig.colorbar(im1, ax=ax1,pad=0.01)
ax1.set_xlabel("$y/\lambda$")
ax1.set_ylabel("$z/\lambda$")
ax1.set_title("log |$E_y^2$|")

im2 = ax2.imshow(np.log10(Int_Ex),cmap=cmap, vmin=-4, extent=extent)
fig.colorbar(im2,ax=ax2,pad=0.01)
ax2.set_xlabel("$y/\lambda$")
ax2.set_ylabel("$z/\lambda$")
ax2.set_title("log |$E_x^2$|")