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

S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/sim_Tp45')
l0=2*np.pi

track_name = "track_eon"

if track_name == "track_eon":
    T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px","Ex","Ez","By","Bz"])
else:
    T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px",])

Ltrans = S.namelist.Ltrans

track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()


N_part = 20


x = track_traj["x"][:,::N_part]
track_N = x.shape[1]

y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]


if track_name == "track_eon":
    Ex = track_traj["Ex"][:,::N_part]
    Ez = track_traj["Ez"][:,::N_part]
    
    # Bx = track_traj["Ex"][:,::N_part]
    # By = track_traj["Ey"][:,::N_part]
    # Bz = track_traj["Ez"][:,::N_part]

r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py

hlx = np.abs(Lx_track[-1]>100)

plt.plot(y[:,hlx],z[:,hlx])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# ax.plot(x[:,149], y[:,149], zs=z[:,149])
ax.plot(x[:,427], y[:,427], zs=z[:,427])
# ax.plot(x[:,447], y[:,447], zs=z[:,447])


plt.figure()
plt.plot(t_range, Ez[:,427])
plt.plot(t_range, Ez[:,427]*0+np.mean(Ez[:,427]))

plt.grid()


