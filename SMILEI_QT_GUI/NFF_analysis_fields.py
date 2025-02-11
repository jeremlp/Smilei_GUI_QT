# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:06:41 2024

@author: Jeremy
"""

import os
import sys
import numpy as np
from numpy import sqrt, exp, sin, cos, arctan2,pi
from scipy import special
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
l0=2*np.pi

plt.close("all")
pwr = 10
c_super=sqrt(2)*special.gamma((pwr+1)/pwr)
c_gauss = sqrt(pi/2)
def sin2(t):
    return np.sin(pi*t/Tp)**2*(t<=Tp)
def gauss(t):
    return np.exp(-((t-Tp)/(Tp/sqrt(2.)/2/c_gauss))**2)
def superGauss(t):
    return np.exp(-((t-Tp)/(Tp/sqrt(2.)/2/c_super))**10)

SAVE = True

base_path = f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NFF/'

sim_list = ["nff_sin2_a2_Tp6_w2.5","nff_sin2_a2_Tp6_w2.5_LEHE","nff_sin2_a2_Tp6_w2.5_BOUCHARD","nff_sin2_a2_Tp6_AMC_dx64","nff_sin2_a2_Tp6_AMC_dx128","nff_Gaussian_a2_Tp6_w2.5","nff_SuperGaussian_a2_Tp6_w2.5"]

t_range_smooth = np.arange(0,30*l0,0.1)

a0 = 2
Tp=6*l0
w0=2.5*l0

fig = plt.figure(figsize=(12,6))
ax1,ax2 = fig.subplots(1,2)
ax1.grid()
ax1.set_yscale("log")
ax1.set_ylim(1e-9,10)
ax1.set_xlabel("t/t0")
ax1.set_ylabel("|Ey|(x=0)")
ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*sin2(t_range_smooth),"k--")
ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*gauss(t_range_smooth),"k--")
ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*superGauss(t_range_smooth),"k--")
ax1.set_title("Transverse field |Ey|")
ax2.grid()
ax2.set_yscale("log")
ax2.set_ylim(1e-10,10)
ax2.set_xlabel("t/t0")
ax2.set_ylabel("|Ex|(x=0)")
ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*sin2(t_range_smooth)/w0,"k--")
ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*gauss(t_range_smooth)/w0,"k--")
ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*superGauss(t_range_smooth)/w0,"k--")
ax2.set_title("Longitudinal field |Ex|")
for sim in sim_list:
    S = happi.Open(base_path+sim)
    w0 = S.namelist.w0
    Tp = S.namelist.Tp
    a0 = S.namelist.a0

    Ey_hd_diag = S.Probe("temp_env","Ey")
    Ey_hd = np.array(Ey_hd_diag.getData())
    abs_Ey_x0 = np.max(np.abs(Ey_hd[:,0,:,:]),axis=(1,2))
    t_range = Ey_hd_diag.getTimes()
    
    Ex_hd_diag = S.Probe("temp_env","Ex")
    Ex_hd = np.array(Ex_hd_diag.getData())
    abs_Ex_x0 = np.max(np.abs(Ex_hd[:,0,:,:]),axis=(1,2))
    
    ax1.plot(t_range/l0,abs_Ey_x0,label=sim)
    ax2.plot(t_range/l0,abs_Ex_x0,label=sim)
    
ax1.legend()
ax2.legend()

fig.suptitle("Impact of temporal envelope shape on non-physical fields\n Probe at x=0")
fig.tight_layout()

if SAVE: fig.savefig(f"{os.environ['SMILEI_QT']}/figures/_OPTICAL_/NFF/nff_probe_temp_env.png")

#============================================================================
# Laser duration effect
#============================================================================
sim_list2 = ["nff_sin2_a2_Tp6_w2.5","nff_sin2_a2_Tp12_w2.5"]

fig = plt.figure(figsize=(12,6))
ax1,ax2 = fig.subplots(1,2)
ax1.grid()
ax1.set_yscale("log")
ax1.set_ylim(1e-10,10)
ax1.set_xlabel("t/t0")
ax1.set_ylabel("|Ey|(x=0)")
ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*sin2(t_range_smooth),"k--")
# ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*gauss(t_range_smooth),"k--")
# ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*superGauss(t_range_smooth),"k--")
ax1.set_title("Transverse field |Ey|")
ax2.grid()
ax2.set_yscale("log")
ax2.set_ylim(1e-10,10)
ax2.set_xlabel("t/t0")
ax2.set_ylabel("|Ex|(x=0)")
ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*sin2(t_range_smooth)/w0,"k--")
# ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*gauss(t_range_smooth)/w0,"k--")
# ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*superGauss(t_range_smooth)/w0,"k--")
ax2.set_title("Longitudinal field |Ex|")
for sim in sim_list2:
    S = happi.Open(base_path+sim)
    w0 = S.namelist.w0
    Tp = S.namelist.Tp
    a0 = S.namelist.a0

    Ey_hd_diag = S.Probe("temp_env","Ey")
    Ey_hd = np.array(Ey_hd_diag.getData())
    abs_Ey_x0 = np.max(np.abs(Ey_hd[:,0,:,:]),axis=(1,2))
    
    Ex_hd_diag = S.Probe("temp_env","Ex")
    Ex_hd = np.array(Ex_hd_diag.getData())
    abs_Ex_x0 = np.max(np.abs(Ex_hd[:,0,:,:]),axis=(1,2))
    
    t_range = Ey_hd_diag.getTimes()
    
    ax1.plot(t_range/l0,abs_Ey_x0,label=sim)
    ax2.plot(t_range/l0,abs_Ex_x0,label=sim)


ax1.legend()
ax2.legend()

fig.suptitle("Impact of pulse duration on non-physical fields\n Probe at x=0")
fig.tight_layout()

if SAVE: fig.savefig(f"{os.environ['SMILEI_QT']}/figures/_OPTICAL_/NFF/nff_probe_sin2_Tp.png")




sim_list2 = ["nff_Gaussian_a2_Tp6_w2.5","nff_Gaussian_a2_Tp12_w2.5"]

fig = plt.figure(figsize=(12,6))
ax1,ax2 = fig.subplots(1,2)
ax1.grid()
ax1.set_yscale("log")
ax1.set_ylim(1e-10,10)
ax1.set_xlabel("t/t0")
ax1.set_ylabel("|Ey|(x=0)")
ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*sin2(t_range_smooth),"k--")
# ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*gauss(t_range_smooth),"k--")
# ax1.plot(t_range_smooth/l0,a0*exp(-0.5)*superGauss(t_range_smooth),"k--")
ax1.set_title("Transverse field |Ey|")
ax2.grid()
ax2.set_yscale("log")
ax2.set_ylim(1e-12,10)
ax2.set_xlabel("t/t0")
ax2.set_ylabel("|Ex|(x=0)")
ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*sin2(t_range_smooth)/w0,"k--")
# ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*gauss(t_range_smooth)/w0,"k--")
# ax2.plot(t_range_smooth/l0,a0*exp(-0.5)*superGauss(t_range_smooth)/w0,"k--")
ax2.set_title("Longitudinal field |Ex|")
for sim in sim_list2:
    S = happi.Open(base_path+sim)
    w0 = S.namelist.w0
    Tp = S.namelist.Tp
    a0 = S.namelist.a0

    Ey_hd_diag = S.Probe("temp_env","Ey")
    Ey_hd = np.array(Ey_hd_diag.getData())
    abs_Ey_x0 = np.max(np.abs(Ey_hd[:,0,:,:]),axis=(1,2))
    
    Ex_hd_diag = S.Probe("temp_env","Ex")
    Ex_hd = np.array(Ex_hd_diag.getData())
    abs_Ex_x0 = np.max(np.abs(Ex_hd[:,0,:,:]),axis=(1,2))
    
    t_range = Ey_hd_diag.getTimes()
    
    ax1.plot(t_range/l0,abs_Ey_x0,label=sim)
    ax2.plot(t_range/l0,abs_Ex_x0,label=sim)


ax1.legend()
ax2.legend()

fig.suptitle("Impact of pulse duration on non-physical fields\n Probe at x=0")
fig.tight_layout()

if SAVE: fig.savefig(f"{os.environ['SMILEI_QT']}/figures/_OPTICAL_/NFF/nff_probe_Gaussian_Tp.png")


