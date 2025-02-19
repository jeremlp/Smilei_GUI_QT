# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:02:01 2025

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
    return np.sin(pi*t/Tp)**2* ((t>0) & (t<Tp))
def gauss(t):
    t_center = 1.25*Tp+5*l0
    return np.exp(-((t-t_center)/(Tp/sqrt(2.)/2/c_gauss))**2)
def superGauss(t):
    return np.exp(-((t-Tp)/(Tp/sqrt(2.)/2/c_super))**10)

t_range_smooth = np.arange(0,40*l0,0.1)


SAVE = False

a0 = 2
Tp=6*l0
w0 = 2.5*l0
x_pos = 5*l0
base_path = f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NFF/'

sim_list = ["nff_sin2_a2_Tp6_w2.5","nff_Gaussian_a2_Tp6_w2.5",
            "nff_SuperGaussian_a2_Tp6_w2.5", "gauss_a1_Tp6_NET_GAIN_dx64_EGOR"
            'gauss_a2_Tp6','gauss_a2_Tp12']



sim_list = ["gauss_a2_Tp6","gauss_a2_Tp12"]





S1 = happi.Open(base_path+"nff_sin2_a2_Tp6_w2.5")
Ey_probe = S1.Probe("fields","Ey")
paxisX = Ey_probe.getAxis("axis1")[:,0]
paxisY = Ey_probe.getAxis("axis2")[:,1] - S1.namelist.Ltrans/2
paxisZ = Ey_probe.getAxis("axis3")[:,2] - S1.namelist.Ltrans/2

Ey_yee = np.array(Ey_probe.getData())

extent = [paxisX[0]/l0,paxisX[-1]/l0, paxisY[0]/l0, paxisY[-1]/l0]
time = 23
mid_idx = len(paxisZ)//2


S2 = happi.Open(base_path+"nff_sin2_a2_Tp6_w2.5_LEHE")
Ey_lehe = np.array(S2.Probe("fields","Ey").getData())

S3 = happi.Open(base_path+"nff_sin2_a2_Tp6_w2.5_BOUCHARD")
Ey_bouchard = np.array(S3.Probe("fields","Ey").getData())

S4 = happi.Open(base_path+"nff_sin2_a2_Tp6_AMC_dx128")
Ey_AMC128 = np.array(S4.Probe("fields","Ey").getData())

S5 = happi.Open(base_path+"nff_sin2_a2_Tp6_AMC_dx64")
Ey_AMC64 = np.array(S5.Probe("fields","Ey").getData())

S6 = happi.Open(base_path+"nff_sin2_a2_Tp6_w2.5_M4")
Ey_M4 = np.array(S6.Probe("fields","Ey").getData())


fig = plt.figure(figsize=(12,6))
ax1,ax2,ax3,ax4,ax5 = fig.subplots(5,1)

VMAX = 5e-7

im1=ax1.imshow(Ey_yee[time,:,:,mid_idx].T,extent=extent,cmap="RdBu",aspect="auto",vmin=-VMAX,vmax=VMAX)
fig.colorbar(im1,ax=ax1,pad=0.01)
ax1.set_title("Yee")
ax1.set_ylabel("$y/\lambda$")

im2=ax2.imshow(Ey_lehe[time,:,:,mid_idx].T,extent=extent,cmap="RdBu",aspect="auto",vmin=-VMAX,vmax=VMAX)
fig.colorbar(im2,ax=ax2,pad=0.01)
ax2.set_title("Lehe")
ax2.set_ylabel("$y/\lambda$")

im3=ax3.imshow(Ey_bouchard[time,:,:,mid_idx].T,extent=extent,cmap="RdBu",aspect="auto",vmin=-VMAX,vmax=VMAX)
fig.colorbar(im3,ax=ax3,pad=0.01)
ax3.set_title("Bouchard")
ax3.set_ylabel("$y/\lambda$")

im4=ax4.imshow(Ey_AMC128[time,:,:,mid_idx].T,extent=extent,cmap="RdBu",aspect="auto",vmin=-VMAX,vmax=VMAX)
fig.colorbar(im4,ax=ax4,pad=0.01)
ax4.set_title("AMC dx=$\lambda/128$ AM=8")
# ax4.set_xlabel("$x/\lambda$")
ax4.set_ylabel("$y/\lambda$")

im5=ax5.imshow(Ey_M4[time,:,:,mid_idx].T,extent=extent,cmap="RdBu",aspect="auto",vmin=-VMAX,vmax=VMAX)
fig.colorbar(im5,ax=ax5,pad=0.01)
ax5.set_title("M4")
ax5.set_xlabel("$x/\lambda$")
ax5.set_ylabel("$y/\lambda$")

t_range = S1.Probe("fields","Ey").getTimes()
ax1.axvline(t_range[time]/l0,ls="--",color="g",lw=2)
ax1.axvline(t_range[time]/l0-Tp/l0,ls="--",color="g",lw=2)
t_range = S2.Probe("fields","Ey").getTimes()
ax2.axvline(t_range[time]/l0,ls="--",color="g",lw=2)
ax2.axvline(t_range[time]/l0-Tp/l0,ls="--",color="g",lw=2)
t_range = S3.Probe("fields","Ey").getTimes()
ax3.axvline(t_range[time]/l0,ls="--",color="g",lw=2)
ax3.axvline(t_range[time]/l0-Tp/l0,ls="--",color="g",lw=2)
t_range = S4.Probe("fields","Ey").getTimes()
ax4.axvline(t_range[time]/l0,ls="--",color="g",lw=2)
ax4.axvline(t_range[time]/l0-Tp/l0,ls="--",color="g",lw=2)
t_range = S6.Probe("fields","Ey").getTimes()
ax5.axvline(t_range[time]/l0,ls="--",color="g",lw=2)
ax5.axvline(t_range[time]/l0-Tp/l0,ls="--",color="g",lw=2)

fig.suptitle("Transverse field E_y for different Maxwell Solver")
# ax1.set_xlim(12,18)
# ax2.set_xlim(12,18)
# ax3.set_xlim(12,18)

fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()