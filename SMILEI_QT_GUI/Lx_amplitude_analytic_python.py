# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:10:10 2025

@author: Jeremy
"""

import sys
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import logging

logging.getLogger('matplotlib.font_manager').disabled = True
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from numpy import sqrt, pi, exp, cos, sin,arctan2
import time
from scipy import integrate
from scipy.interpolate import griddata
import math
# from numba import njit
l0 = 2*pi
t0 = l0
plt.close("all")



a0=3.0
w0=2.5*l0
Tp=12*l0
Tint = Tp*3/8
x0=5*l0
x_pos =x0
z =x0

l1=1


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

def LxEpolar_V2_O3(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 +
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) -
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 +
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) +
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr
def LxEpolar_V2_O3(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O3(r,theta,z)

def LxEpolar_V2_O5(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O5(r,theta,z)

# @njit
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




r_range = np.arange(0,2*w0,0.02)
theta_range = np.arange(0,2*pi,pi/32)
R,THETA = np.meshgrid(r_range,theta_range)

r1 = w0*sqrt(2* (2 - sqrt(2)))/2
r2_x = sqrt(4*w0**4 + 8*x0**2 + 2*sqrt(2)*sqrt(w0**8 + 4*w0**4*x0**2 + 8*x0**4)) / (w0 * 2)
r1_x = sqrt(4*w0**4 + 8*x0**2 - 2*sqrt(2)*sqrt(w0**8 + 4*w0**4*x0**2 + 8*x0**4)) / (w0 * 2)


COEF = sqrt(1+(a0*f(r_range,x0))**2+ 1/4*(a0*f(r_range,x0))**4)
Lx_max_model = np.max(COEF*LxEpolar_V2_O5(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
plt.plot(r_range/l0,Lx_max_model,"k-", lw=2,label="x0=5*l0 O5")

Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,x0))**2+ 1/4*(a0*f(r_range,x0))**4)
plt.plot(r_range/l0,COEF*Lx_max_model,"r--", lw=2,label="x0=5*l0 O3")


Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,0))**2+ 1/4*(a0*f(r_range,0))**4)
plt.plot(r_range/l0,COEF*Lx_max_model,"g--", lw=2,label="x0=0")

# plt.plot(r_range/l0,-COEF*Lx_max_model,"g-", lw=2, label="Model $\gamma$$L_z^{NR}$")

# plt.plot(r_range/l0,-COEF*Lx_max_model,"k--", lw=2, label="Model $\gamma$$L_z^{NR}$")



# plt.axvline(r1/l0,color="k",ls="--")
plt.axvline(r1_x/l0,color="k",ls="--")
# plt.axvline(r2_x/l0,color="k",ls="--")
# plt.axvline(9.2/l0,color="g",ls="--")

theta = pi/4
expr_x = (a0**2 * exp((-2*w0**4 - 4*z**2 + sqrt(2) * sqrt(w0**8 + 4*w0**4*z**2 + 8*z**4)) / (w0**4 + 4*z**2)) 
              * Tint * w0**2 
              * (2*w0**4 + 4*z**2 - sqrt(2) * sqrt(w0**8 + 4*w0**4*z**2 + 8*z**4)) 
              * (4*w0**2*z*cos(2*theta) + (4*z**2 - sqrt(2) * sqrt(w0**8 + 4*w0**4*z**2 + 8*z**4)) * sin(2*theta))
             ) / (2 * (w0**4 + 4*z**2)**3)

expr_rmin = (np.sqrt(2) - 1) * np.exp(np.sqrt(2) - 2) * (a0**2 * Tint / w0**2)

COEF_rmin = sqrt(1+(a0*f(r1,0))**2+ 1/4*(a0*f(r1,0))**4)
COEF_rmin_x0 = sqrt(1+(a0*f(r1,x0))**2+ 1/4*(a0*f(r1,x0))**4)


plt.axhline(COEF_rmin*expr_rmin,color="g",ls="--",label="Analytical O3 (x=0)")
plt.axhline(COEF_rmin_x0*expr_rmin,color="orange",ls="--",label="Analytical O3 (x=x0)")


plt.axhline(COEF_rmin_x0*LxEpolar_V2_O3(r1,-pi/4,0,w0,a0,3/8*Tp),color="k",ls="--",label="LxO3 (x=x0)")
plt.axhline(COEF_rmin_x0*LxEpolar_V2_O5(r1,-pi/4,0,w0,a0,3/8*Tp),color="gray",ls="--",label="LxO5 (x=x0)")

plt.grid()
plt.legend()
