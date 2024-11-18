# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:13:12 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy.interpolate import splrep, splev
import scipy

S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_PHYSICAL/sim_OAM_Long')
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/plasma_OAM_XDIR_NO_IONS_DIAGS')

l0=2*np.pi

track_name = "track_eon_full"

if track_name == "track_eon":
    T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px","Ex","Ey","Ez","By","By","Bz"])
else:
    T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px",])

Ltrans = S.namelist.Ltrans

track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()


N_part = 1


x = track_traj["x"][:,::N_part]
track_N = x.shape[1]

y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)

if track_name == "track_eon":
    Ex = track_traj["Ex"][:,::N_part]
    Ey = track_traj["Ey"][:,::N_part]
    Ez = track_traj["Ez"][:,::N_part]
    
    Bx = track_traj["Ex"][:,::N_part]
    By = track_traj["Ey"][:,::N_part]
    Bz = track_traj["Ez"][:,::N_part]
    
    E_theta = (y*Ez - z*Ey)/r

Lx_track =  y*pz - z*py

plt.figure()
plt.scatter(y[0]/l0,z[0]/l0,c=px[-1],cmap="RdYlBu",vmin=-0.5,vmax=0.5,s=1)
plt.colorbar()

plt.figure()
bins = np.linspace(-3,3,200)
plt.hist(px[-1],bins=bins,edgecolor="k", linewidth=1.2,alpha=0.33)
plt.hist(py[-1],bins=bins,edgecolor="k", linewidth=1.2,alpha=0.33)
plt.hist(pz[-1],bins=bins,edgecolor="k", linewidth=1.2,alpha=0.33)
plt.grid()

zaeeaeaaezaz

hlx = np.abs(Lx_track[-1]>80)

id_hlx = np.where(hlx==True)[0]

plt.plot(y[:,hlx]/l0,z[:,hlx]/l0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for id in id_hlx:
    ax.plot(x[:,id]/l0, y[:,id]/l0, zs=z[:,id]/l0,label=id)
plt.plot((0,160),(0,0),(0,0),"k--",alpha=0.75)
plt.legend()

# plt.figure()
# plt.plot(y[:,661]/l0,z[:,661]/l0)
# plt.grid()

plt.figure()
plt.plot(t_range/l0, E_theta[:,661])
plt.plot(t_range/l0, E_theta[:,661]*0+np.mean(E_theta[:,661]))

plt.grid()


