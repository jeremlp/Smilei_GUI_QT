# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:53:09 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy.optimize import curve_fit

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt

sim_loc_list = ["SIM_OPTICAL_NR_HD/opt_base_PML_dx64",
                "SIM_OPTICAL_A0.3_HD/opt_a0.3_dx48",
                "SIM_OPTICAL_A1_HD/opt_a1.0_dx64",
                "SIM_OPTICAL_A1.5_HD/opt_a1.5_dx48",
                "SIM_OPTICAL_A2_HD/opt_a2.0_dx64",
                "SIM_OPTICAL_A2.33_HD/opt_a2.33_dx48",
                "SIM_OPTICAL_A2.5_HD/opt_a2.5_dx48",
                "SIM_OPTICAL_A3_HD/opt_a3.0_dx32"]
a0_range = np.array([0.1,0.3,1,1.5,2,2.33,2.5,3])

Lx_amplitude_list = []

N_part = 5
for k,sim_loc in enumerate(sim_loc_list):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc}')
    T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez"])

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
    
    plt.scatter(r[0],Lx_track[-1],s=1,alpha=0.25,label=f"a0={a0_range[k]}")
    Lx_amplitude_list.append(np.nanmax(np.abs(Lx_track[-1])))
    

def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr

a0_range_smooth = np.arange(0.1,3.5,0.01)

def func_powerlaw(x, m, c):
    return x**m * c

target_func = func_powerlaw

popt, pcov = curve_fit(target_func, a0_range, Lx_amplitude_list)


r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R, THETA = np.meshgrid(r_range,theta_range)

Tint = 3/8*Tp
z0=5*l0
Lx_max_model_list = []
for a0 in a0_range_smooth:
    Lx_max_model = np.max(LxEpolar(R,THETA,z0,w0,a0,Tint))
    Lx_max_model_list.append(Lx_max_model)
Lx_max_model_list = np.array(Lx_max_model_list)

Lx_max_model_list_gamma = np.sqrt(1+a0_range_smooth**2)*Lx_max_model_list

plt.grid()
plt.legend()

plt.figure()
plt.plot(a0_range_smooth,Lx_max_model_list,"k-",label="Model")
plt.plot(a0_range_smooth,Lx_max_model_list_gamma,"r--",label="Model*gamma")
plt.plot(a0_range,Lx_amplitude_list,"o",c="C0",label="Smilei")

plt.grid()
# m,c = popt
# plt.plot(a0_range_smooth, target_func(a0_range_smooth, *popt), '--',label=f"$a_0^{{{m:.3f}}}$")
# plt.plot(a0_range_smooth, c*a0_range_smooth**2*np.sqrt(1+a0_range_smooth**2),"--",label="$a_0^2\cdot\sqrt{1+a_0^2}$")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("a0")
plt.title("Lx amplitude scaling with a0")
plt.ylabel("max |Lx|")
plt.tight_layout()