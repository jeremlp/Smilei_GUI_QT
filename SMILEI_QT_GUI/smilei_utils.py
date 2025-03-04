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
l1 = 1
def SET_CONSTANTS(w0_value, a0_value, Tp_value, l1_value = 1):
    global w0, a0,Tp, l1
    w0 = w0_value
    a0 = a0_value
    Tp = Tp_value
    l1 = l1_value


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
        std_Lx[i] = np.std(Y[mask])/np.sqrt(len(Y[mask]))
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