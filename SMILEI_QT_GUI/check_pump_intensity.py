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


diag = S.Probe("Ey_intensity","Ey")
Ey = np.array(diag.getData()).astype(np.float32)

print(Ey.shape)
time = 10
Int = Ey**2


x_range = diag.getAxis("axis1")[:,0]/l0


W = 15*3

arr_cumsum = np.cumsum(Int,axis=1)
result = (arr_cumsum[:,W:] - arr_cumsum[:,:-W]) / W
print(result)


plt.figure()
plt.imshow(result[-1,-1], cmap="viridis",aspect="auto")
plt.colorbar()
plt.title("Intensity <Ey^2> transverse distribution")
plt.tight_layout()

plt.figure()
plt.imshow(result[-1,:,:,100].T, cmap="viridis",aspect="auto")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Intensity <Ey^2> longitudinal distribution")
plt.tight_layout()