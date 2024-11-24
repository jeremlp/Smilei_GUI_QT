# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:50:18 2024

@author: jerem
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_OPTICAL_A2_HD/opt_a2.0_dx48')
l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt


w0 = S.namelist.w0
Tp = S.namelist.Tp
a0 = S.namelist.a0


def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr

    
track_name = "track_eon"

if track_name == "track_eon":
    T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px","Ex","Ey","Ez","By","By","Bz"])
else:
    T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px",])

Ltrans = S.namelist.Ltrans

track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1


x = track_traj["x"][:,::N_part]
track_N = x.shape[1]

y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)
Lx_track =  y*pz - z*py

sidx = x[0]<13*l0
midx = (x[0]<23*l0) & (x[0]>17*l0)
eidx = x[0]>27*l0

plt.scatter(r[0,sidx]/l0,Lx_track[-1,sidx],s=1,alpha=0.5,label="close")
# plt.scatter(r[0,midx]/l0,Lx_track[-1,midx],s=1,alpha=0.5,label="mid")
# plt.scatter(r[0,eidx]/l0,Lx_track[-1,eidx],s=1,alpha=0.5,label="far")

# plt.colorbar()
plt.grid()

r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,0.01)
R_grid, Theta_grid = np.meshgrid(r_range,theta_range)
z_foc_lz = 5*l0
Tint = 3/8*Tp
Lx2_model = np.max(LxEpolar(R_grid,Theta_grid,z_foc_lz,w0,a0,Tint),axis=0)
plt.plot(r_range/l0,Lx2_model,"k--",alpha=1)
plt.plot(r_range/l0,-Lx2_model,"k--",alpha=1, label="Model $L_z^{(2)}$")
plt.legend()


"""
CHECK EFFECTIVE PULSE DURATION

Ex = track_traj["Ex"]
Ey = track_traj["Ey"]

plt.figure()
plt.plot(t_range/l0-5,np.max(np.abs(Ey),axis=-1))
plt.grid()

Ey_pulse = np.max(np.abs(Ey),axis=-1)
dt_diag = (t_range[1]-t_range[0])/l0
Tp_eff1 = np.max(np.diff(np.where(Ey_pulse<0.01*np.max(Ey_pulse))[0]))*dt_diag
Tp_eff2 = 2*(t_range[np.where(Ey_pulse==np.max(Ey_pulse))[0][0]]/l0-5)
print(Tp_eff1,Tp_eff2)

"""
