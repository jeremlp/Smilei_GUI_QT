# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:10:30 2024

@author: jerem
"""

import os
import sys
import numpy as np
from numpy import sqrt, exp, sin, cos, arctan2,pi
from scipy import special
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
l0=2*np.pi
plt.close("all")
import matplotlib.font_manager

SAVE = False

base_path = f'{os.environ["SMILEI_CLUSTER"]}'

sim_list = ["nff_sin2_a2_Tp6_w2.5",'gauss_a2_Tp6']

fig = plt.figure(figsize=(12,6))
ax1,ax2 = fig.subplots(2)

fig2 = plt.figure(figsize=(12,6))
axa,axb = fig2.subplots(2)

VMAX = 1e-5
t_idx = -14
S = happi.Open(f"{base_path}/nff_sin2_a2_Tp6_w2.5")
Ey_diag = S.Probe("fields","Ey")
x_axis = Ey_diag.getAxis("axis1")[:,0]
y_axis = Ey_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2
z_axis = Ey_diag.getAxis("axis3")[:,2] - S.namelist.Ltrans/2
t_range = Ey_diag.getTimes()
Ey = np.array(Ey_diag.getData())
middle_trans = Ey.shape[-1]//2
extent = [x_axis[0]/l0,x_axis[90]/l0, y_axis[0]/l0, y_axis[-1]/l0]
im = ax1.imshow(Ey[t_idx,:90,:,middle_trans].T,cmap="RdBu",aspect="auto", vmin=-VMAX, vmax=VMAX, extent=extent)
fig.colorbar(im,ax=ax1,pad=0.01)
ax1.set_title(f"Ey for Sin2 envelope at t={t_range[t_idx]/l0:.2f}")

im = axa.imshow(np.log10(np.abs(Ey[t_idx,:90,:,middle_trans].T)),cmap="jet",aspect="auto", vmin=-6, vmax=1, extent=extent)
fig2.colorbar(im,ax=axa,pad=0.01)
axa.set_title(f"log|Ey| for Sin2 envelope at t={t_range[t_idx]/l0:.2f}")

t_idx = -12
S = happi.Open(f"{base_path}/gauss_a2_Tp6")
Ey_diag = S.Probe("fields","Ey")
x_axis = Ey_diag.getAxis("axis1")[:,0]
y_axis = Ey_diag.getAxis("axis2")[:,1]- S.namelist.Ltrans/2
z_axis = Ey_diag.getAxis("axis3")[:,2]- S.namelist.Ltrans/2
t_range = Ey_diag.getTimes()
Ey = np.array(Ey_diag.getData())
middle_trans = Ey.shape[-1]//2
extent = [x_axis[0]/l0,x_axis[90]/l0, y_axis[0]/l0, y_axis[-1]/l0]

im = ax2.imshow(Ey[t_idx,:90,:,middle_trans].T,cmap="RdBu",aspect="auto", vmin=-VMAX, vmax=VMAX, extent=extent)
fig.colorbar(im,ax=ax2,pad=0.01)
ax2.set_title(f"Ey for Gaussian envelope at t={t_range[t_idx]/l0:.2f}")

im = axb.imshow(np.log10(np.abs(Ey[t_idx,:90,:,middle_trans].T)),cmap="jet",aspect="auto", vmin=-6, vmax=1, extent=extent)
fig2.colorbar(im,ax=axb,pad=0.01)
axb.set_title(f"log|Ey| for Gaussian envelope at t={t_range[t_idx]/l0:.2f}")


ax1.set_xlabel("$x/\lambda$")
ax1.set_ylabel("$y/\lambda$")
ax2.set_xlabel("$x/\lambda$")
ax2.set_ylabel("$y/\lambda$")
fig.tight_layout()
fig2.tight_layout()



axa.set_xlabel("$x/\lambda$")
axa.set_ylabel("$y/\lambda$")
axb.set_xlabel("$x/\lambda$")
axb.set_ylabel("$y/\lambda$")
fig2.tight_layout()