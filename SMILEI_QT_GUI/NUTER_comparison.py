# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:52:46 2024

@author: Jeremy
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open(r'C:\Users\Jeremy\_LULI_\CLUSTER\NUTER\NUTER_xfoc10')
l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt

plt.close("all")
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

    
track_name = "track_eon_dense"


T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px"])
Ltrans = S.namelist.Ltrans

track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()

N_part = 10
x = track_traj["x"][:,::N_part]
track_N = x.shape[1]

y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(y,x)
Lx_track =  y*pz - z*py

idx = (x[0]<100*l0)
idx2 = (x[0]>=5*l0) & (x[0]<6*l0)
idx3 = (x[0]>=12*l0) 

print(t_range[-1])

l1=1
import math
def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def f_prime(r,z):
    C_lp = np.sqrt(1/math.factorial(abs(l1)))
    return C_lp/w(z)**3 * exp(-(r/w(z))**2) * (r/w(z))**(abs(l1)-1) * (-2*r**2+w(z)**2*abs(l1))
def f_squared_prime(r,z):
    return 2*w0**2/(w(z)**2*r) * f(r,z)**2*(abs(l1)-2*(r/w0)**2+ 4*(z**2/w0**4))

r_range = np.arange(w0/sqrt(2),2*w0,0.1)
# plt.plot(r_range,-1/4*f(r_range,0)**2 *(abs(l1)-2*(r_range/w0)**2)*t_range[-1]**2,"r--")

p_end = -a0**2/4*f_squared_prime(r_range, -10*l0)*3/8*Tp
dr = p_end*(t_range[-1]-Tp)

plt.figure()
plt.scatter(r[0,idx],r[-1,idx]-r[0,idx],c=Lx_track[-1],s=1,cmap="RdBu")
plt.colorbar()
# plt.scatter(r[0,idx2],r[-1,idx2]-r[0,idx2],s=1)
# plt.scatter(r[0,idx3],r[-1,idx3]-r[0,idx3],s=1)

plt.plot(r_range,dr,"r-")
plt.xlabel("r0")
plt.ylabel("r-r0")
plt.title("Radial displacement")

# p_end = -a0**2/4*f_squared_prime(r_range, 5*l0-10*l0)*3/8*Tp
# dr = p_end*(t_range[-1]-Tp-5*l0)
# plt.plot(r_range,dr,"-",color="blue")
# plt.grid()

# p_end = -a0**2/4*f_squared_prime(r_range, -10*l0)*3/8*Tp
# dr = p_end*(t_range[-1]-Tp)
# plt.plot(r_range,dr,"-",c="purple")
# plt.axvline(w0/sqrt(2),color="k")

aezaezaezeaz

plt.figure()
plt.scatter(x[0]/l0,y[0]/l0,s=1)
plt.xlabel("X0/λ")
plt.ylabel("Y0/λ")
plt.title("Initial longitudinal distribution of electrons")
plt.grid()
plt.figure()
plt.scatter(y[0]/l0,z[0]/l0,s=1)
plt.xlabel("Y0/λ")
plt.ylabel("Z0/λ")
plt.title("Initial transverse distribution of electrons")
plt.grid()

track_name = "track_eon"


T0 = S.TrackParticles(track_name, axes=["x","y","z","py","pz","px"])
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
theta = np.arctan2(z,y)
Lx_track =  y*pz - z*py

idx = (x[0]<100*l0)
idx2 = (x[0]>=5*l0) & (x[0]>6*l0)


fig = plt.figure(figsize=(10,6))
axs = fig.subplots(2,2)
(ax1,ax2,ax3,ax4) = axs[0,0],axs[0,1],axs[1,0],axs[1,1]
axs_list = [ax1,ax2,ax3,ax4]

delay = np.mean(x[0])/l0

Nidx = np.where(( np.abs(r[0]-13.9)<0.25) & (np.abs(np.rad2deg(theta[0])+150)<0.2))[0][0]
ax1.plot(t_range/l0-delay,Lx_track[:,Nidx])
ax1.set_title("Lx")
ax2.plot(t_range/l0-delay,r[:,Nidx])
ax2.set_title("r")
ax3.plot(t_range/l0-delay,x[:,Nidx])
ax3.set_title("x")
ax4.plot(t_range/l0-delay,np.rad2deg(theta[:,Nidx]))
ax4.set_title("theta [deg]")
fig.tight_layout()


for ax in axs_list:
    ax.grid()
    ax.set_xlim(0,7)
    
    
ax1.set_ylim(-11,11)
ax2.set_ylim(12,20)
ax3.set_ylim(72,80)
ax4.set_ylim(-156,-146)
    