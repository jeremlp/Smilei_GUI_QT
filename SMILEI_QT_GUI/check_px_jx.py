# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:58:40 2024

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


Ltrans = S.namelist.Ltrans

# diag = S.ParticleBinning("px_r")
# data = np.array(diag.getData())
# plt.figure()
# plt.plot(data[-1])
# # plt.colorbar()
# plt.title("px_r")
# plt.grid()

diag = S.ParticleBinning("px_r_W")
dataW = np.array(diag.getData())
plt.figure()
plt.plot(dataW[::15].T*137*3.125)
# plt.colorbar()
plt.title("px_r_W")
plt.grid()

#=======================================================================

Bvx_long = S.ParticleBinning("jx")
jx = np.mean(np.array(Bvx_long.getData()),axis=-1)
plt.figure()
plt.imshow(-jx[-1].T, cmap="RdYlBu", aspect="auto",vmin=-0.01, vmax=0.01)
plt.colorbar()
plt.title("-jx")  


# diag = S.ParticleBinning("px_r_x_W")
# data = np.array(diag.getData())
# plt.figure()
# plt.imshow(data[-1], cmap="RdYlBu", aspect="auto",vmin=-2*10**-5, vmax=2*10**-5)
# plt.colorbar()
# plt.title("px_r_x_W")

plt.figure()
Bptheta_long = S.ParticleBinning("px_W")
px = np.mean(np.array(Bptheta_long.getData()),axis=-1)
plt.imshow(px[-1].T, cmap="RdYlBu", aspect="auto",vmin=-0.01, vmax=0.01)

plt.colorbar()
plt.title("px")  

plt.figure()
# plt.plot(np.mean(data[-1],axis=1))
plt.plot(np.mean(px[-1],axis=0))

plt.grid()



plt.figure()
bins = np.linspace(-0.05,0.05,75)
plt.hist(px[-1].ravel(),bins=bins,ec="k",lw=1.15)