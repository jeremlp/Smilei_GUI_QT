# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:59:54 2024

@author: jerem
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/plasma_OAM_XDIR_NO_IONS_DIAGS')

l0=2*np.pi

ne = 0.03

a = np.array(S.ParticleBinning("weight_z_x").getData()[-1])
b = np.array(S.ParticleBinning("weight_y_x").getData()[-1])

plt.figure()
plt.imshow(a,cmap="jet",vmin=0,vmax=0.03, aspect="auto")
plt.colorbar()
plt.figure()
plt.imshow(b,cmap="jet",vmin=0,vmax=0.03, aspect="auto")
plt.colorbar()

