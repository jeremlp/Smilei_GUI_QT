# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:09 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_OPTICAL_A1_HD/opt_a1.0_dx48_CUSTOM_3')
S2 = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_OPTICAL_A1_HD/opt_a1.0_dx48_CUSTOM_4')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt


T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","By","By","Bz"])
T02 = S2.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","By","By","Bz"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
track_N_tot = T0.nParticles
t_range = T0.getTimes()
t_range2 = T02.getTimes()

track_traj = T0.getData()
track_traj2 = T02.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1


x = track_traj["x"][:,::N_part]
track_N = x.shape[1]

py = track_traj["py"][:,::N_part]
Ex = track_traj["Ex"]
Ey = track_traj["Ey"]

py2 = track_traj2["py"][:,::N_part]
Ex2 = track_traj2["Ex"]
Ey2 = track_traj2["Ey"]

plt.figure()
plt.plot(t_range,np.gradient(Ey[:,0]),label="d Ey")
plt.plot(t_range,np.gradient(py[:,0]),label="d py")
plt.plot(t_range2,np.gradient(Ey2[:,0]),"--",label="d Ey_CUSTOM")
plt.plot(t_range2,np.gradient(py2[:,0]),"--",label="d py_CUSTOM")
plt.axhline(1e-3,color="k",ls="--")
plt.axhline(-1e-3,color="k",ls="--")
plt.legend()
plt.grid()

plt.figure()
plt.plot(t_range/l0-5, np.max(Ey,axis=-1),label='Ey')
plt.plot(t_range2/l0-5, np.max(Ey2,axis=-1),"--",label='Ey')
plt.plot(t_range/l0-5, np.max(Ex,axis=-1),label='Ex_CUSTOM')
plt.plot(t_range2/l0-5, np.max(Ex2,axis=-1),"--",label='Ex_CUSTOM')

plt.plot(t_range/l0,np.sin(pi*t_range/Tp)**2*(t_range<Tp),"--k")
plt.axhline(1e-3,color="k",ls="--")
plt.grid()
plt.legend()
plt.yscale("log")
plt.ylim(10**-6,1)




