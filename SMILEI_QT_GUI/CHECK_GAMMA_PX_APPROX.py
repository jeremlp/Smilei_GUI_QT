# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:36:36 2025

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
from scipy.signal import savgol_filter

plt.close("all")

l0=2*pi

sim = "gauss_a3_Tp12_dx128_AM8"
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}\SIM_OPTICAL_GAUSSIAN\{sim}')

T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1

track_N_tot = T0.nParticles
t_range = T0.getTimes()
print(f"{t_range=}")
# print("Every:",S.namelist.DiagTrackParticles[2].every)

track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1
    
if "AM" in sim:
    track_r_center = 0
else:
    track_r_center = Ltrans/2

x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
x_pos = x[0,0]


Nid_mid = np.where((r[0] > 1*l0) & (r[0]<2*l0))[0]


plt.figure()
plt.scatter(r[0,Nid_mid]/l0, Lx_track[-1,Nid_mid],c= np.max(np.abs(1-(gamma-px)),axis=0)[Nid_mid],s=1,cmap="jet")
plt.colorbar()
plt.xlabel("$r_0/\lambda$",fontsize=12)
plt.ylabel("$L_x$",fontsize=12)
plt.title(f"$L_x$ distribution with $\gamma - p_x$ error\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
plt.grid()



Nid_HL = np.where(np.abs(Lx_track[-2]) > 0.5)[0]

def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)

x0=5*l0


gamma_mean = sqrt(1+(f(r[0,Nid_HL],x0)*a0)**2/2 + 1/16*(f(r[0,Nid_HL],x0)*a0)**4)[0]
gamma_max = sqrt(1+(f(r[0,Nid_HL],x0)*a0)**2 + 1/4*(f(r[0,Nid_HL],x0)*a0)**4)[0]

gamma_slow = savgol_filter(gamma[:,Nid_mid], window_length=501,polyorder=2)


plt.figure()
# plt.plot(t_range/l0, gamma[:,Nid_mid])
plt.plot(t_range/l0, gamma_slow)

plt.axhline(gamma_max,label="$\gamma_{max}$", ls="--",color="r")
plt.axhline(gamma_mean,label="$<\gamma>$",ls="--",color="k")
plt.grid()
