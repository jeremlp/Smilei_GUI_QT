# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:59:54 2024

@author: jerem
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('D:/JLP/CMI/_MASTER_2_/_STAGE_LULI_/CLUSTER/plasma_OAM_XDIR_NO_IONS_DIAGS')
l0=2*np.pi

def radial_profile(data):
    center = (data.shape[0]//2,data.shape[1]//2)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

ne = 0.03

Ltrans = S.namelist.Ltrans/l0
a = np.array(S.ParticleBinning("weight_yz").getData()[-1])/ne

weight = np.mean(np.array(S.ParticleBinning("weight_av").getData()[-1])/ne,axis=-1)
idx = round(weight.shape[0]*0.25)

c = np.mean(weight[:-2*idx],axis=0)
# plt.figure()
# plt.imshow(a,cmap="jet",vmin=0,vmax=3, aspect="auto")
# plt.colorbar()
# plt.figure()
# plt.imshow(b,cmap="jet",vmin=0,vmax=3, aspect="auto")
# plt.colorbar()

plt.figure()
plt.plot(np.linspace(-Ltrans/2,Ltrans/2,len(c)),c)

h = radial_profile(a)
plt.plot(np.linspace(0,Ltrans/2*np.sqrt(2),len(h)),h)