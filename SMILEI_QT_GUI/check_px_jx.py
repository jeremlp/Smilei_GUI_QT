# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:58:40 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi

t0 = time.perf_counter()

S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/plasma_OAM_XDIR_NO_IONS_DIAGS')
# S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_PHYSICAL/sim_OAM_Long')

       

l0=2*np.pi


Ltrans = S.namelist.Ltrans



"""
px_scalar, px_r, px_y, px_z, px_y_x_W, px_r_x_W, px_W
"""
"""


diag = S.ParticleBinning("px_r")
dataW = np.array(diag.getData())
plt.figure()
plt.plot(dataW[-1].T*137*3.125)
plt.title("px_r")
plt.grid()

diag = S.ParticleBinning("px_y")
dataY = np.array(diag.getData())
diag = S.ParticleBinning("px_z")
dataZ = np.array(diag.getData())

plt.figure()
plt.plot(dataY[-1].T*137*3.125)
plt.plot(dataZ[-1].T*137*3.125)

plt.title("px_y / px_z")
plt.grid()

#=======================================================================

# Bvx_long = S.ParticleBinning("jx")
# jx = np.mean(np.array(Bvx_long.getData()),axis=-1)
# plt.figure()
# plt.imshow(-jx[-1].T, cmap="RdYlBu", aspect="auto",vmin=-0.01, vmax=0.01)
# plt.colorbar()
# plt.title("-jx")  


diag = S.ParticleBinning("px_r_x_W")
data = np.array(diag.getData())
plt.figure()
plt.imshow(data[-1], cmap="RdYlBu", aspect="auto",vmin=-2*10**-5, vmax=2*10**-5)
plt.colorbar()
plt.title("px_r_x_W")
"""
t_idx = -1
plt.figure()
Bptheta_long = S.ParticleBinning("2D_jx")
jx = np.mean(np.array(Bptheta_long.getData()),axis=-1)
plt.imshow(-jx[t_idx].T, cmap="RdYlBu", aspect="auto",vmin=-0.01, vmax=0.01)
plt.colorbar()
plt.title("jx")  

# t_idx = -1
# plt.figure()
# jx[np.abs(jx)>0.001] = np.nan

# plt.imshow(-jx[t_idx].T, cmap="RdYlBu", aspect="auto",vmin=-0.01, vmax=0.01)
# plt.colorbar()
# plt.title("jx")  




S.ParticleBinning("px_scalar").plot



plt.figure()
Bpx_long = S.ParticleBinning("2D_px")
px = np.mean(np.array(Bpx_long.getData()),axis=-1)
plt.imshow(px[t_idx,:,:].T, cmap="RdYlBu", aspect="auto",vmin=-10, vmax=10)
plt.colorbar()
plt.title("px")  

plt.figure()
plt.imshow(px[t_idx].T, cmap="jet", aspect="auto",vmin=-10, vmax=0)
plt.colorbar()
plt.title("jx")  


# plt.figure()
# # plt.plot(np.mean(data[-1],axis=1))
# plt.plot(np.mean(-jx[t_idx],axis=0))

# plt.grid()

diag_p = S.ParticleBinning("p")
px_range_av = diag_p.getAxis("p")
data_p = np.array(diag_p.getData())

diag = S.ParticleBinning("px")
px_range = diag.getAxis("px")
data = np.array(diag.getData())

diag = S.ParticleBinning("px_av")
px_range2 = diag.getAxis("user_function0")
data2 = np.array(diag.getData())

plt.figure()

plt.plot(px_range,data[t_idx]/0.03,label="Integrated Binning")
# plt.plot(px_range2,data2[t_idx]/0.03,label="Integrated Binning Weight")


bins = np.linspace(-10,10,200)
# plt.hist(px[t_idx].ravel(),bins=bins,lw=1.1, histtype='step',density=True,label="px")
plt.yscale("log")
plt.hist(px[t_idx,:,:].ravel(),bins=bins,lw=1.1, histtype='step',density=True,label="px")
# plt.hist(px[t_idx,0:150].ravel(),bins=bins,lw=1.1, histtype='step',density=True,label="px s")
# plt.hist(px[t_idx,300:350].ravel(),bins=bins,lw=1.1, histtype='step',density=True,label="px e")
plt.hist(px[t_idx,450:].ravel(),bins=bins,lw=1.1, histtype='step',density=True,label="px")

plt.legend()
plt.grid()