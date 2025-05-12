# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:59:25 2025

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
import time
from tqdm import tqdm

w0 = None
a0 = None
Tp = None
eps,l1 = 0,1
zR = None

def SET_CONSTANTS(w0_value, a0_value, Tp_value, l1_value = 1):
    global w0, a0,Tp, l1
    w0 = w0_value
    a0 = a0_value
    Tp = Tp_value
    l1 = l1_value
    
    # global zR, eps,Tp, l1, C_lp, t_center, c_gauss, sigma_gauss
    # zR = 0.5*w0**2
    # eps=0
    # C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
    # t_center=1.25*Tp
    # c_gauss = sqrt(pi/2)
    # sigma_gauss = Tp*3/8/c_gauss


#===================================================
# POST-PROCESS FUNCTIONS
#===================================================
def averageAM(X,Y,dr_av, da = 0.04):
    # da = 0.04
    t0 = time.perf_counter()
    print("Computing average...",da)
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    av_Lx = np.empty(a_range.shape)
    std_Lx = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        av_Lx[i] = np.nanmean(Y[mask])
        std_Lx[i] = np.nanstd(Y[mask])/np.sqrt(len(Y[mask]))
    t1 = time.perf_counter()
    print(f"...{(t1-t0):.0f} s")
    return a_range,av_Lx, std_Lx

def medianAM(X,Y,dr_av):
    da = 0.04
    t0 = time.perf_counter()
    print("Computing average...",da)
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    med_Lx = np.empty(a_range.shape)
    std_Lx = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        med_Lx[i] = np.nanmedian(Y[mask])
        std_Lx[i] = np.std(Y[mask])/np.sqrt(len(Y[mask]))
    t1 = time.perf_counter()
    print(f"...{(t1-t0):.0f} s")
    return a_range,med_Lx, std_Lx

def min_max(X,Y,dr_av=0.6, da=0.04):
    M = []
    m = []
    # da = 0.05
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    M = np.empty(a_range.shape)
    m = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        M[i] = np.nanmax(Y[mask])
        m[i] = np.nanmin(Y[mask])
    return a_range,m,M

def min_max_percentile(X,Y,dr_av=0.6, percentile=93, da=0.04):
    M = []
    m = []
    # da = 0.05
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    M = np.empty(a_range.shape)
    m = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        M[i] = np.nanpercentile(Y[mask],percentile)
        m[i] = np.nanpercentile(Y[mask],100-percentile)
    return a_range,m,M


#===================================================
# LAGUERRE-GAUSSIAN FUNCTIONS
#===================================================

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


def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)

def gauss2_int(t,x0):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x0)/sigma_gauss)+1)
def gauss2_int_int(t,x):
    t_center = 1.25*Tp
    psi = lambda t,x : sqrt(2)*(t-t_center-x)/sigma_gauss
    expr =lambda t,x : 0.5*sqrt(pi/2)*sigma_gauss*( gauss(t,x)**2*sigma_gauss/sqrt(2*pi) + t + (t-t_center-x)*(special.erf(psi(t,x))))
    return expr(t,x) - expr(0,x)



#===================================================
# LAGUERRE-GAUSSIAN FUNCTIONS
#===================================================

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


#===================================================
# PONDEROMOTIVE FUNCTIONS
#===================================================
def dx_func(r,theta,x,t):
    """ use 1/gamma ?"""
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return 1/gamma * a0**2/4 * f(r,x)**2 * gauss2_int(t,x) 

def dtheta_func(r,theta,x,t):
    """ possible 1/r missing for theta velocity"""
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1/gamma * 1/r * Ftheta_V2_O5(r,theta,x) * gauss2_int_int(t, x)

def theta_func(r,theta,x,t):
    r_model = np.abs(r + dr_func(r,x,t))
    x_model = x + dx_func(r, theta, x, t)

    # idx_cross = np.where(r_model==np.min(r_model))[0][0]
    theta0_m = np.zeros_like(r_model) + theta
    # theta0_m[idx_cross:] = pi+theta0_m[idx_cross:] #Allow model to cross beam axis and switch theta

    y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(t,x_model)*cos(l1*theta0_m  - t + x_model)
    z_model = r_model*sin(theta0_m)

    theta_model = np.arctan2(z_model, y_model) + dtheta_func(r, theta0_m, x_model, t)

    return theta_model

def dr_func_small_r(r,theta,x,t):
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1/gamma**2 *a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_func_large_r(r,theta,x,t):
    """For large r, using 1/gamma works better than the true 1/gamma^2 """
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1/gamma *a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_func_test_r(r,theta,x,t):
    """For test r, using 1 """
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1 *a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_func(r,x,t):
    return dr_func_large_r(r,theta,x,t)