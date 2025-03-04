# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:44:54 2025

@author: Jeremy
"""

import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
import time
from tqdm import tqdm
plt.close("all")

l0=2*pi

a0_requested = 2.5
sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3.5_Tp12_dx128_AM4",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4.5_Tp12_dx128_AM4"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,3.5,4,4.5])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

N_part = 1
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])


Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()
if "AM" in sim_path:
    track_r_center = 0
else:
    track_r_center = Ltrans/2
    
x = track_traj["x"][:,::N_part]
track_N = x.shape[1]
y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]
r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
p_theta = p_theta = Lx_track/r
theta = np.arctan2(z,y)

x0 = x[0,0]



def min_max(X,Y,dr_av=0.15):
    M = []
    m = []
    da = 0.05
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    M = np.empty(a_range.shape)
    m = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        M[i] = np.nanmax(Y[mask])
        m[i] = np.nanmin(Y[mask])
    return a_range,m,M

fig, ax = plt.subplots(1)

a_range, min_Lx, max_Lx = min_max(r[0],Lx_track[-1],dr_av=0.3)
ax.plot(a_range/l0,max_Lx,"C1")
ax.plot(a_range/l0,min_Lx,"C2")
ax.grid()
ax.set_xlabel("$r_0/\lambda$")
ax.set_ylabel("$L_x$")
ax.set_title("$L_x(r)$ at $a_0=4$")

fig2, ax2 = plt.subplots(1)
scat = ax2.scatter(y[0]/l0, z[0]/l0, c=Lx_track[-1], cmap="RdYlBu",s=2)
ax2.grid()
fig2.colorbar(scat, ax=ax2)
ax2.set_xlabel("$y_0/\lambda$")
ax2.set_ylabel("$z_0/\lambda$")
ax2.set_title("$L_x(y,z)$ of electrons after the interaction")
requested_r0 = 1.4*l0




Nid_r = np.where( np.abs(r[0]-requested_r0) < 0.02)[0]
r = r[:,Nid_r]
Lx_track = Lx_track[:,Nid_r]
x = x[:,Nid_r]
y = y[:,Nid_r]
z = z[:,Nid_r]

Nidp = np.where(Lx_track[-1] == np.max(Lx_track[-1]))[0]
Nidm = np.where(Lx_track[-1] == np.min(Lx_track[-1]))[0]

ax.scatter(r[0,Nidp]/l0, Lx_track[-1,Nidp],s=20,marker="x",color="k")
ax.scatter(r[0,Nidm]/l0, Lx_track[-1,Nidm],s=20,marker="x",color="k")
ax2.scatter(y[0,Nidm]/l0, z[0,Nidm]/l0,s=20,marker="x",color="k")
ax2.scatter(y[0,Nidp]/l0, z[0,Nidp]/l0,s=20,marker="x",color="k")

plt.figure()

plt.plot(y[:-400,Nidm]/l0, z[:-400,Nidm]/l0,color="C1",label="$L_{x}^-$")
plt.plot(y[:-400,Nidp]/l0, z[:-400,Nidp]/l0,color="C2",label="$L_{x}^+$")
plt.scatter(y[0,Nidm]/l0, z[0,Nidm]/l0,s=20,marker="x",color="k",zorder=10000)
plt.scatter(y[0,Nidp]/l0, z[0,Nidp]/l0,s=20,marker="x",color="k",zorder=10000)
plt.scatter(0, 0,s=200,marker="x",color="r")
plt.grid()
plt.legend()
plt.xlabel("$y_0/\lambda$")
plt.ylabel("$z_0/\lambda$")
plt.title("Trajectories in transverse plane of 2 electrons")


plt.figure()

plt.plot(t_range/l0, Lx_track[:,Nidm],color="C1",label="$L_{x}^-$")
plt.plot(t_range/l0, Lx_track[:,Nidp],color="C2",label="$L_{x}^+$")

plt.grid()
plt.legend()
plt.xlabel("$t/(c\lambda)$")
plt.ylabel("$L_x$")
plt.title("$L_x(t)$ function of time for 2 electrons")
