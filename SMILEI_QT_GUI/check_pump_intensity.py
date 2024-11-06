# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:52:27 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/plasma_OAM_XDIR_NO_IONS_DIAGS')
l0=2*np.pi


diag = S.Probe("Exy_intensity","Ey")
Ey = np.array(diag.getData()).astype(np.float32)

print(Ey.shape)

middle_trans = Ey.shape[-1]//2

time = 10
Int = Ey**2


x_range = diag.getAxis("axis1")[:,0]/l0
y_range = diag.getAxis("axis2")[:,1]/l0 - S.namelist.Ltrans/2
t_range = diag.getTimes()


W = 10

arr_cumsum = np.cumsum(Int,axis=1)
result = (arr_cumsum[:,W:] - arr_cumsum[:,:-W]) / W




plt.figure()
plt.imshow(result[-1,:,:,middle_trans].T, cmap="jet",aspect="auto")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Intensity <Ey^2> longitudinal distribution")
plt.tight_layout()


ne = 0.03
Tp = 30*l0

time_idx = -1

empirical_corr = 0.98 #to compensate for other effect on top of dis
groupe_velocity = empirical_corr*1/np.sqrt(ne+1) #c^2k/sqrt(wp^2+c^2k^2)
laser_x_pos_range = np.max([groupe_velocity*t_range-Tp/2,t_range*0],axis=0)/l0

xcut_idx_range = []
arr = np.empty((len(t_range),123))

for i,laser_x_pos in enumerate(laser_x_pos_range):
    xcut_idx = np.where(np.abs(x_range-laser_x_pos) == np.min(np.abs(x_range-laser_x_pos)))[0][0]
    xcut_idx_range.append(xcut_idx)
    arr[i] = result[i,xcut_idx,:,middle_trans]

time_extent = [t_range[0]/l0, t_range[-1]/l0,y_range[0]/l0,y_range[-1]/l0]
plt.figure()
plt.imshow(arr.T, cmap="jet",aspect="auto",extent=time_extent)
plt.colorbar()
plt.xlabel("t")
plt.ylabel("y")
plt.title("Intensity <Ey^2> maximum pulse center distribution function of time")
plt.tight_layout()

max_intensity = np.max(result[:,:,:,middle_trans],axis=(1,2))
x_idx_max = []
for t in range(len(t_range)):
    x_idx_max.append(np.where(result[t,:,:,middle_trans]==max_intensity[t])[0][0])

print(max_intensity)

plt.figure()
plt.plot(xcut_idx_range)
plt.plot(x_idx_max)
plt.grid()

