# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:42:46 2024

@author: Jeremy
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy.optimize import curve_fit
plt.close("all")
l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt

sim_loc_list_Tp = ["SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6",
                   "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp9_dx32",
                   "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12",
                   "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp16_dx32"]


Tp_range = np.array([6,9,12,16])*l0

Lx_amplitude_list_Tp = []
N_part = 1
for k,sim_loc_Tp in enumerate(sim_loc_list_Tp):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_Tp}')
    T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])

    Ltrans = S.namelist.Ltrans
    Tp = S.namelist.Tp
    w0 = S.namelist.w0
    track_N_tot = T0.nParticles
    t_range = T0.getTimes()
    
    track_traj = T0.getData()

    print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

    x = track_traj["x"][:,::N_part]
    track_N = x.shape[1]
    
    y = track_traj["y"][:,::N_part]-Ltrans/2
    z = track_traj["z"][:,::N_part] -Ltrans/2
    py = track_traj["py"][:,::N_part]
    pz = track_traj["pz"][:,::N_part]
    px = track_traj["px"][:,::N_part]
    r = np.sqrt(y**2+z**2)
    Lx_track =  y*pz - z*py
    
    # plt.scatter(r[0],Lx_track[-1],s=1,alpha=0.25,label=f"a0={a0_range[k]}")
    Lx_amplitude_list_Tp.append(np.nanmax(np.abs(Lx_track[-1])))

def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr

Tp_range_smooth = np.arange(3,18,0.1)*l0


r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R, THETA = np.meshgrid(r_range,theta_range)

z0=5*l0
Lx_max_model_list_Tp = []


for Tp in Tp_range_smooth:
    Lx_max_model_Tp = np.max(LxEpolar(R,THETA,z0,w0,a0=2,Tint=3/8*Tp))
    Lx_max_model_list_Tp.append(Lx_max_model_Tp)
Lx_max_model_list_Tp = np.array(Lx_max_model_list_Tp)




# plt.grid()
# plt.legend()

plt.figure()
k = exp(-0.5)
a0=2.0
gamma = np.sqrt(1 + (k*a0)**2 + 1/4*(k*a0)**4)
plt.plot(Tp_range_smooth/l0,Lx_max_model_list_Tp*gamma,"k-",label="Model", lw=2)
# m,b = np.polyfit(Tp_range/l0, Lx_amplitude_list_Tp, 1)

def func(x, a, b):
    return a*x**b  # a and d are redundant
popt, pcov = curve_fit(func, Tp_range, Lx_amplitude_list_Tp)


plt.plot(Tp_range_smooth/l0,func(Tp_range_smooth, *popt),"r--",label=f"fit {popt[0]:.3f}*x^{popt[1]:.2f}", lw=2)

plt.plot(Tp_range/l0,Lx_amplitude_list_Tp,"o",c="C0",label="Smilei",markersize=10)
plt.grid()
plt.legend()
plt.xlabel("$T_p/t_0$")
plt.title(f"Lx amplitude scaling with Tp\na0=2; w0={w0/l0:.1f}λ")
plt.ylabel("max |Lx|")
# plt.xlim(1.45,4.15)
plt.tight_layout()
plt.pause(0.1)
