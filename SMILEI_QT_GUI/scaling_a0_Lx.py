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
plt.close("all")
l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt

Tp_requested = 6
sim_loc_list_16 = ["SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp16_dx32",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp16_dx32"]

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx32",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3.5_Tp12_dx32",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx32"]

sim_loc_list_9 = ["SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp9_dx32",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp9_dx32"]

sim_loc_list_6 = ["SIM_OPTICAL_NR_HD/opt_base_PML_dx64",
                "SIM_OPTICAL_A0.3_HD/opt_a0.3_dx48",
                "SIM_OPTICAL_A1_HD/opt_a1.0_dx64",
                "SIM_OPTICAL_A1.5_HD/opt_a1.5_dx48",
                "SIM_OPTICAL_A2_HD/opt_a2.0_dx64",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6",
                "SIM_OPTICAL_A2.33_HD/opt_a2.33_dx48",
                "SIM_OPTICAL_A2.5_HD/opt_a2.5_dx48",
                "SIM_OPTICAL_A3_HD/opt_a3.0_dx32",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp6_dx32"]

a0_range_6 =  np.array([0.1,0.3,1,1.5,2,2,2.33,2.5,3,4])
a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,3.5,4])
a0_range_16 = np.array([2,3])
a0_range_9 =  np.array([2,3])

if Tp_requested==6:
    sim_loc_list = sim_loc_list_6

elif Tp_requested==12:
    sim_loc_list = sim_loc_list_12
    a0_range = a0_range_6
else:
    sim_loc_list = sim_loc_list_6
    a0_range = a0_range_12

Lx_amplitude_list_6 = []
N_part = 1
for k,sim_loc_6 in enumerate(sim_loc_list_6):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_6}')
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
    Lx_amplitude_list_6.append(np.nanmax(np.abs(Lx_track[-1])))
    
Lx_amplitude_list_12 = []
for k,sim_loc_12 in enumerate(sim_loc_list_12):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_12}')
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
    Lx_amplitude_list_12.append(np.nanmax(np.abs(Lx_track[-1])))

Lx_amplitude_list_9 = []
for k,sim_loc_9 in enumerate(sim_loc_list_9):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_9}')
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
    Lx_amplitude_list_9.append(np.nanmax(np.abs(Lx_track[-1])))
    
Lx_amplitude_list_16 = []
for k,sim_loc_16 in enumerate(sim_loc_list_16):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_16}')
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
    Lx_amplitude_list_16.append(np.nanmax(np.abs(Lx_track[-1])))

def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr

a0_range_smooth = np.arange(0.1,4+0.1,0.01)


r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R, THETA = np.meshgrid(r_range,theta_range)

z0=5*l0
Lx_max_model_list_6 = []
Lx_max_model_list_12 = []
Lx_max_model_list_16 = []
Lx_max_model_list_9 = []

for a0 in a0_range_smooth:
    Lx_max_model_6 = np.max(LxEpolar(R,THETA,z0,w0,a0,3/8*6*l0))
    Lx_max_model_list_6.append(Lx_max_model_6)
    Lx_max_model_12 = np.max(LxEpolar(R,THETA,z0,w0,a0,3/8*12*l0))
    Lx_max_model_list_12.append(Lx_max_model_12)
    Lx_max_model_9 = np.max(LxEpolar(R,THETA,z0,w0,a0,3/8*9*l0))
    Lx_max_model_list_9.append(Lx_max_model_9)
    Lx_max_model_16 = np.max(LxEpolar(R,THETA,z0,w0,a0,3/8*16*l0))
    Lx_max_model_list_16.append(Lx_max_model_16)
    
Lx_max_model_list_6 = np.array(Lx_max_model_list_6)
Lx_max_model_list_12 = np.array(Lx_max_model_list_12)
Lx_max_model_list_16 = np.array(Lx_max_model_list_16)
Lx_max_model_list_9 = np.array(Lx_max_model_list_9)



# plt.grid()
# plt.legend()

plt.figure()
plt.plot(a0_range_smooth,Lx_max_model_list_6,"k-",label="Model", lw=2)
k = exp(-0.5)
a0r = a0_range_smooth
plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1+a0r**2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1+a0r**2/2),
         "--",color="C1",label="Model*$\sqrt{1+a_0^2/2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1 + k*a0r**2 + 1/4*(k*a0r)**4),
          "-", color="C2",label="Model*$\sqrt{1+0.6*(a_0)^2 + 1/4(0.6*a0)^4}$", lw=3)
plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1 + (k*a0r)**2 + 1/4*(k*a0r)**4),
          "-", color="r",label="Model*$\sqrt{1+(0.6*a_0)^2 + 1/4(0.6*a0)^4}$", lw=2)


# plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1 + (a0r)**2/2 + 1/4*1/4*(a0r)**4),
#           "--", color="purple",label="Model*$\sqrt{1+(0.6*a_0)^2 + 1/4(0.6*a0)^4}$", lw=2)
# plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1 + k*a0r**2),
#          "-", color="C2",label="Model*$\sqrt{1+0.6*(a_0)^2}$", lw=3)
# plt.plot(a0_range_smooth,Lx_max_model_list_6 * np.sqrt(1 + (k*a0r)**2 ),
#          "-", color="r",label="Model*$\sqrt{1+(0.6*a_0)^2}$", lw=2)


plt.plot(a0_range_6,Lx_amplitude_list_6,"o",c="C0",label="Smilei",markersize=10)
plt.grid()
plt.legend()
plt.xlabel("a0")
plt.title(f"Lx amplitude scaling with a0\nTp=6$~t_0;$ w0={w0/l0:.1f}λ")
plt.ylabel("max |Lx|")
plt.xlim(1.45,4.15)
plt.tight_layout()
plt.pause(0.1)

plt.figure()
plt.plot(a0_range_smooth,Lx_max_model_list_12,"k-",label="Model", lw=2)
k = exp(-0.5)
a0r = a0_range_smooth
plt.plot(a0_range_smooth,Lx_max_model_list_12 * np.sqrt(1+a0r**2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_12 * np.sqrt(1+a0r**2/2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2/2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_12 * np.sqrt(1 + k*a0r**2 + 1/4*(k*a0r)**4),
         "-", color="C2",label="Model*$\sqrt{1+0.6*(a_0)^2 + 1/4(0.6*a0)^4}$", lw=3)

plt.plot(a0_range_smooth,Lx_max_model_list_12 * np.sqrt(1 + (k*a0r)**2 + 1/4*(k*a0r)**4),
         "-", color="r",label="Model*$\sqrt{1+(0.6*a_0)^2 + 1/4(0.6*a0)^4}$", lw=2)
plt.plot(a0_range_12,Lx_amplitude_list_12,"o",c="C0",label="Smilei",markersize=10)
plt.grid()
plt.legend()
plt.xlabel("a0")
plt.title(f"Lx amplitude scaling with a0\nTp=12$~t_0;$ w0={w0/l0:.1f}λ")
plt.ylabel("max |Lx|")
plt.xlim(1.45,4.15)
plt.tight_layout()



plt.figure()
plt.plot(a0_range_smooth,Lx_max_model_list_9,"k-",label="Model", lw=2)
k = exp(-0.5)
a0r = a0_range_smooth
plt.plot(a0_range_smooth,Lx_max_model_list_9 * np.sqrt(1+a0r**2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_9 * np.sqrt(1+a0r**2/2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2/2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_9 * np.sqrt(1 + k*a0r**2 + 1/4*(k*a0r)**4),
         "-", color="C2",label="Model*$\sqrt{1+0.6*(a_0)^2 + 1/4(0.6*a0)^4}$", lw=3)

plt.plot(a0_range_smooth,Lx_max_model_list_9 * np.sqrt(1 + (k*a0r)**2 + 1/4*(k*a0r)**4),
         "-", color="r",label="Model*$\sqrt{1+(0.6*a_0)^2 + 1/4(0.6*a0)^4}$", lw=2)
plt.plot(a0_range_9,Lx_amplitude_list_9,"o",c="C0",label="Smilei",markersize=10)
plt.grid()
plt.legend()
plt.xlabel("a0")
plt.title(f"Lx amplitude scaling with a0\nTp=9$~t_0;$ w0={w0/l0:.1f}λ")
plt.ylabel("max |Lx|")
plt.xlim(1.45,4.15)
plt.tight_layout()


plt.figure()
plt.plot(a0_range_smooth,Lx_max_model_list_16,"k-",label="Model", lw=2)
k = exp(-0.5)
a0r = a0_range_smooth
plt.plot(a0_range_smooth,Lx_max_model_list_16 * np.sqrt(1+a0r**2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_16 * np.sqrt(1+a0r**2/2),
         "-.",color="C1",label="Model*$\sqrt{1+a_0^2/2}$", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_16 * np.sqrt(1 + k*a0r**2 + 1/4*(k*a0r)**4),
         "-", color="C2",label="Model*$\sqrt{1+0.6*(a_0)^2 + 1/4(0.6*a0)^4}$", lw=3)

plt.plot(a0_range_smooth,Lx_max_model_list_16 * np.sqrt(1 + (k*a0r)**2 + 1/4*(k*a0r)**4),
         "-", color="r",label="Model*$\sqrt{1+(0.6*a_0)^2 + 1/4(0.6*a0)^4}$", lw=2)
plt.plot(a0_range_16,Lx_amplitude_list_16,"o",c="C0",label="Smilei",markersize=10)
plt.grid()
plt.legend()
plt.xlabel("a0")
plt.title(f"Lx amplitude scaling with a0\nTp=16$~t_0;$ w0={w0/l0:.1f}λ")
plt.ylabel("max |Lx|")
plt.xlim(1.45,4.15)
plt.tight_layout()