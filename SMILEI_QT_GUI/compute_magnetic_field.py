# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:39:21 2024

@author: jerem
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('D:/JLP/CMI/_MASTER_2_/_STAGE_LULI_/CLUSTER/SIM_PHYSICAL/sim_Tp45')
l0=2*np.pi

Bptheta_trans = S.ParticleBinning("ptheta_W_trans")
ptheta = np.array(Bptheta_trans.getData())

plt.imshow(ptheta[-1,2], cmap="RdYlBu", aspect="auto",vmin=-0.005,vmax=0.005)
plt.colorbar()


y_range = Bptheta_trans.getAxis("y") - S.namelist.Ltrans/2
z_range = Bptheta_trans.getAxis("z") - S.namelist.Ltrans/2

Y,Z = np.meshgrid(y_range, z_range)
Y = Y.T
Z = Z.T
R = np.sqrt(Y**2+Z**2)
THETA = np.arctan2(Z,Y)

print(R.shape,THETA.shape)


plt.figure()
plt.imshow(R*ptheta[-1,2], cmap="RdYlBu", aspect="auto",vmin=-0.05,vmax=0.05)
plt.colorbar()

df_dy = np.gradient(R*ptheta[-1,2], y_range[1] - y_range[0],axis=0) * np.cos(THETA)
df_dz = np.gradient(R*ptheta[-1,2], z_range[1] - z_range[0],axis=1) * np.sin(THETA)

e = -1
Bx = e*1/R*(df_dy+df_dz)

plt.figure()
plt.imshow(Bx, cmap="RdYlBu", aspect="auto",vmin=-0.001,vmax=0.001)
plt.colorbar()