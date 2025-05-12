# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:04:31 2025

@author: Jeremy
"""


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
from smilei_utils import *


from scipy.optimize import curve_fit
plt.close("all")

l0 = 2*pi

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3.5_Tp12_dx128_AM4",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4.5_Tp12_dx128_AM4"]


a0_range = np.array([1.0, 2.0, 2.33, 2.5, 3.0, 3.5])

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,3.5,4,4.5])
Lx_amplitude_list_12 = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/amplitude_smilei/Lx_amplitude_list_12.txt")

a0_range_smooth = np.arange(0,4,0.1)

w0 = 2.5*l0
l1=1
def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)


Lx_NT_list = []
Lx_R_smilei_list = []
Lx_R_model_list = []

Lx_tot_list = []
for a0 in a0_range:
    SET_CONSTANTS(w0, a0, Tp)
    Lx_NT = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/new_term_num_smilei/new_term_a{a0}_coefPI.txt")
    Lx_R_smilei = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/new_term_num_smilei/old_term_a{a0}.txt")
    Lx_R_model = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/new_term_num_smilei/old_term_model_a{a0}.txt")

    R = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/new_term_num_smilei/R_a{a0}_coefPI.txt")
    

    a_range, _, max_Lx_NT = min_max_percentile(R,Lx_NT, dr_av=0.22,percentile=95)
    a_range, _, max_Lx_R_smilei = min_max_percentile(R,Lx_R_smilei, dr_av=0.22,percentile=95)
    a_range, _, max_Lx_R_model = min_max_percentile(R,Lx_R_model, dr_av=0.22,percentile=95)

    a_range, _, max_Lx_tot = min_max_percentile(R,Lx_R_smilei+Lx_NT, dr_av=0.22,percentile=95)

    Lx_NT_list.append(np.nanmax(np.abs(max_Lx_NT)))
    Lx_R_smilei_list.append(np.nanmax(np.abs(max_Lx_R_smilei)))
    Lx_R_model_list.append(np.nanmax(np.abs(max_Lx_R_model)))

    Lx_tot_list.append(np.nanmax(np.abs(max_Lx_tot)))
    # Lx_NT_list.append(np.nanmax(np.abs(Lx_NT)))
    # Lx_R_list.append(np.nanmax(np.abs(Lx_R_smilei)))


Lx_NT_list = np.array(Lx_NT_list)
Lx_R_smilei_list = np.array(Lx_R_smilei_list)
Lx_R_model_list = np.array(Lx_R_model_list)

Lx_tot_list = np.array(Lx_tot_list)

plt.figure()

CF = 1#f(1.7*l0,5*l0)

cPerp = (CF*a0_range)**4/4
cPara = (1+(0.6*a0_range_12)**2/2)*(a0_range_12)**2
cPara3 = (CF*a0_range)**2 #*(1+(CF*a0_range)**2/4)
cPara4 = (a0_range)**2.2


plt.plot(a0_range, Lx_NT_list,"o",markersize = 10,label="$L_x^{\perp}$")
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.xlabel("$a_0$",fontsize=12)
plt.ylabel("max $L_x^{\perp}$",fontsize=12)
plt.title("Scaling of $L_x^{\perp}$")

def func(x, a, b):
    return x**b/a
    
popt, pcov = curve_fit(func, a0_range, Lx_NT_list, p0=[150,4])

plt.plot(a0_range, func(a0_range, *popt), 'k--',
         label=f'fit: $a_0$^({popt[-1]:.3f} $\pm$ {np.sqrt(pcov[1,1]):.2f})')
plt.legend()

plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(a0_range_12, Lx_amplitude_list_12/cPara,"o",markersize=10,label="$L_x^{\parallel}$ smilei")

Tint = 3/8*12*l0
r1 = w0*sqrt(2* (2 - sqrt(2)))/2
expr_rmin = (np.sqrt(2) - 1) * np.exp(np.sqrt(2) - 2) * (a0_range_smooth**2 * Tint / w0**2)
COEF2 = 1+(a0_range_smooth*f(r1,0))**2/4
COEF = 1+(a0_range_smooth*f(r1,0))**2/2

# plt.plot(a0_range_smooth,COEF*expr_rmin,"C4--",label="Analytical $L_{x,max}$ with $\gamma_{max}(x=0)$")

plt.plot(a0_range_smooth, expr_rmin/a0_range_smooth**2,"k-",label="Analytical $L_{x,max}$")

plt.plot(a0_range, Lx_NT_list/cPerp ,"o",markersize = 8,label="$L_x^{\perp}$")
plt.plot(a0_range, Lx_R_smilei_list/cPara3 ,"o",markersize = 8,label="$L_x^{\parallel}$ smilei")
plt.plot(a0_range, Lx_R_model_list/cPara4 ,"o",markersize = 8,label="$L_x^{\parallel}$ model")

# plt.yscale("log")
# plt.xscale("log")
plt.grid()
plt.xlabel("$a_0$",fontsize=12)
plt.ylabel("max $L_x$",fontsize=12)
plt.title("Scaling of $L_x$ normalized by $a_0$ scaling")
plt.legend()

