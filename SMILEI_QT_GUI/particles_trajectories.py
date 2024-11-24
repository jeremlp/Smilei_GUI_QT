# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:57:50 2024

@author: jerem
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, exp, sqrt, pi, arctan2
import sys

module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_PHYSICAL/sim_SAM_Long')
l0 = 2*np.pi

T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])
track_traj = T0.getData()
track_t_range = T0.getTimes()
N_part = 10
Ltrans = S.namelist.Ltrans
y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py
del track_traj


vmax = 0.5*np.max(Lx_track)
start = 800
plt.scatter(0,0, marker="x")
plt.scatter(y[start:,::10],z[start:,::10],c=Lx_track[start:,::10],cmap="RdBu",s=2, vmin=-vmax, vmax=vmax)
plt.colorbar()
plt.grid()


a0 = 10
w0 = 6*l0
Tp = 30*l0
Tint = 3/8*Tp
def Lz2LGpolar(r,Theta,z):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr

r_range = np.arange(0,2*w0,l0/64)
theta_range = np.arange(0,2*pi,0.005)

R_grid, Theta_grid = np.meshgrid(r_range,theta_range)

x_grid = R_grid*cos(Theta_grid)
y_grid = R_grid*sin(Theta_grid)

Lz = Lz2LGpolar(R_grid, Theta_grid,0*l0)
plt.scatter(R_grid,Lz,cmap="RdBu",s=1)

plt.figure()
plt.scatter(r[0]/l0,Lx_track[-1],s=1)
plt.scatter(R_grid/l0,Lz,s=1, alpha=0.25)

plt.grid()
