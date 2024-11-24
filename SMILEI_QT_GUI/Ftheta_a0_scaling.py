# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:04:16 2024

@author: jerem
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_OPTICAL_A2_HD/opt_a2.0_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt

T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","By","By","Bz"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0


track_N_tot = T0.nParticles
t_range = T0.getTimes()

track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1



x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)
Lx_track =  y*pz - z*py

tmax = 120
Nid = 0


def Ftheta(r,theta,z):
    #-1 bcs Ftheta uses IntBz instead of IntEz, so need to inverse sign
    expr = -1 * -((exp(-((2 * w0**2 * (r**2 * cos(theta)**2 + r**2 * sin(theta)**2)) / (w0**4 + 4 * z**2))) *
          a0**2 * w0**6 * (-8 * r**2 * w0**2 * cos(theta) * sin(theta) + 
                          4 * (2 * r**2 * z * cos(theta)**2 + 
                               2 * r**4 * cos(theta)**3 * sin(theta) - 
                               2 * r**2 * z * sin(theta)**2 + 
                               2 * r**4 * cos(theta) * sin(theta)**3))) / 
         (2 * r * (w0**4 + 4 * z**2)**3))
    return expr

def FthetaCart(x,y,z):
    #-1 bcs Ftheta uses IntBz instead of IntEz, so need to inverse sign

    r = sqrt(x**2+y**2)
    theta = arctan2(y,x)
    expr = -1 * -((exp(-((2 * w0**2 * (r**2 * cos(theta)**2 + r**2 * sin(theta)**2)) / (w0**4 + 4 * z**2))) *
          a0**2 * w0**6 * (-8 * r**2 * w0**2 * cos(theta) * sin(theta) + 
                          4 * (2 * r**2 * z * cos(theta)**2 + 
                               2 * r**4 * cos(theta)**3 * sin(theta) - 
                               2 * r**2 * z * sin(theta)**2 + 
                               2 * r**4 * cos(theta) * sin(theta)**3))) / 
         (2 * r * (w0**4 + 4 * z**2)**3))
    return expr

Nid = np.where(np.abs(Lx_track[-1])==np.max(np.abs(Lx_track[-1])))[0][0]

z0 = np.mean(z[0])
x_range = np.arange(-2*w0,2*w0,0.1)
X,Y = np.meshgrid(x_range,x_range)
extent = [x_range[0]/l0,x_range[-1]/l0,x_range[0]/l0,x_range[-1]/l0]

plt.figure()
plt.scatter(r[0]/l0,Lx_track[-1],s=1)
plt.axvline(w0/l0,color="k",ls="--",label="w0")
plt.axvline(w0/sqrt(2)/l0,color="red",ls="--",label="w0/sqrt(2)")
plt.grid()
plt.legend()

plt.figure()
plt.imshow(FthetaCart(X,Y,z0),extent=extent,cmap="RdBu")
plt.plot(y[:tmax,Nid]/l0,z[:tmax,Nid]/l0,"k")
plt.scatter(y[0,Nid]/l0,z[0,Nid]/l0,marker="o",color="k")

max_intensity = plt.Circle((0,0),w0/sqrt(2)/l0,fill=False,ec="red")
plt.gca().add_patch(max_intensity)
plt.scatter(0,0,marker="x",color="red")
plt.grid()
plt.tight_layout()




