# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:29:22 2025

@author: Jeremy
"""



import numpy as np

import matplotlib.pyplot as plt
from numpy import pi, cos, sin, arctan2, exp, sqrt
import math
l0=2*pi

import os
import sys
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy import integrate,special
import time
plt.close("all")

def averageAM(X,Y,dr_av):
    da = 0.04
    t0 = time.perf_counter()
    print("Computing average...",da)
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    av_Lx = np.empty(a_range.shape)
    std_Lx = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        av_Lx[i] = np.nanmean(Y[mask])
        std_Lx[i] = np.std(Y[mask])/np.sqrt(len(Y[mask]))
    t1 = time.perf_counter()
    print(f"...{(t1-t0):.0f} s")
    return a_range,av_Lx, std_Lx

def averageModified(X,Y,dr_av):
    t0 = time.perf_counter()
    print("Computing average...")
    a_range = r_range_net
    M = np.empty(a_range.shape)
    STD = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        M[i] = np.nanmean(Y[mask])
        STD[i] = np.std(Y[mask])/np.sqrt(len(Y[mask]))
    t1 = time.perf_counter()
    print(f"...{(t1-t0):.0f} s")
    return a_range,M, STD
def min_max(X,Y,dr_av=0.6):
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
    


sim_loc_list_12 = ["gauss_a3_Tp12_NET_GAIN_dx32_AM4",
                    "gauss_a3_Tp12_NET_GAIN_dx64_AM4",
                    "gauss_a3_Tp12_NET_GAIN_dx128_AM4",
                    "gauss_a3_Tp12_NET_GAIN_dx128_AM8",
                    "gauss_a3_Tp12_NET_GAIN_dx256_AM8",
                    "gauss_a3_Tp12_NET_GAIN_TIME_dx64_AM4"]
dx_range_12 = ["dx=$\lambda/32$, AM=4",
                "dx=$\lambda/64$, AM=4",
                "dx=$\lambda/128$, AM=4",
                "dx=$\lambda/128$, AM=8",
                "dx=$\lambda/256$, AM=8",
                "dx=$\lambda/64$, AM=4, $t_{end}+5\lambda$"]


sim_loc_list_12 = ["gauss_a2_Tp6_NET_GAIN_dx32",
                   "gauss_a2_Tp6_NET_GAIN_dx48",
                    "gauss_a2_Tp6_NET_GAIN_dx64",
                    "gauss_a2_Tp6_NET_GAIN_dx128_AM4",
                    "gauss_a2_Tp6_NET_GAIN_dx128_AM8",
                    "gauss_a2_Tp6_NET_GAIN_dx128_AM16",
                    "gauss_a2_Tp6_NET_GAIN_dx256_AM8"]
dx_range_12 = ["dx=$\lambda/32$",
                "dx=$\lambda/48$",
                "dx=$\lambda/64$",
                "dx=$\lambda/128$, AM=4",
                "dx=$\lambda/128$, AM=8",
                "dx=$\lambda/128$, AM=16", 
                "dx=$\lambda/256$, AM=8"]

# sim_loc_list_12 = ["gauss_a2_Tp12_NET_GAIN_dx64_AM4",
#                     "gauss_a2.5_Tp12_NET_GAIN_dx64_AM4",
#                     "gauss_a3_Tp12_NET_GAIN_dx64_AM4",
#                     "gauss_a3.5_Tp12_NET_GAIN_dx64_AM4",
#                     "gauss_a4_Tp12_NET_GAIN_dx64_AM4"]
# dx_range_12 = ["a0=2",
#                 "a0=2.5",
#                 "a0=3.0",
#                 "a0=3.5",
#                 "a0=4"]


# sim_loc_list_12 = [ "gauss_a3_Tp12_NET_GAIN_dx64_AM4",
#                    "gauss_a3_Tp12_NET_GAIN_dx128_AM4_L-1",
#                     "gauss_a3.5_Tp12_NET_GAIN_dx64_AM4",
#                     "gauss_a4_Tp12_NET_GAIN_dx64_AM4"]
# dx_range_12 = [ "a0=3.0",
#                 "a0=3.0, l=-1"
#                 "a0=3.5",
#                 "a0=4"]

#gauss_a3_Tp12_NET_GAIN_PHASE4_dx64
#"PHASE +PI/4 3D",

# sim_loc_list_12 = ["gauss_a3_Tp12_NET_GAIN_PHASE2_dx64_AM4",
#                     "gauss_a3_Tp12_NET_GAIN_PHASE4_dx64_AM4",
#                     "gauss_a3_Tp12_NET_GAIN_PHASE4_dx64",
#                     "gauss_a3_Tp12_NET_GAIN_dx64_AM4"]

# dx_range_12 = ["PHASE +PI/2",
#                 "PHASE +PI/4",
#                 "PHASE +PI/4 3D",
#                 "PHASE 0"]


Tp_requested = 12

dx_range = dx_range_12
sim_loc_list = sim_loc_list_12


S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/{sim_loc_list[0]}')
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1

# fig1, ax1 = plt.subplots(1)
# ax1.grid()
# ax1.set_xlabel("$r_0/\lambda$")
# ax1.set_title(f"Lx radial distribution\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
# ax1.set_ylabel("Lx")
    
fig2, ax2 = plt.subplots(1)
ax2.set_xlabel("$r_0/\lambda$", fontsize=16)
ax2.grid()
ax2.axvline(w0/sqrt(2)/l0,color="r",ls="--",label="I_max")
ax2.set_title(f"Average $<L_x>$ (normalized) radial distribution\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
# ax2.set_title(f"Mean <Lx> (normalized) radial distribution\n($Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
ax2.set_ylabel("$<L_x>$", fontsize=16)
fig2.tight_layout()
    
plt.pause(0.1)
    


mean_arr = []
k=0
for sim, dx_label in zip(sim_loc_list,dx_range):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/{sim}')
    
    T0 = S.TrackParticles("track_eon_net", axes=["x","y","z","py","pz","px"])
    
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
    
    r_range_net = S.namelist.r_range_net
    
    # ax1.scatter(r[0]/l0,Lx_track[-1],s=1,alpha = 0.5, label=f"dx=$\lambda$/{dx_value}")
    # ax1.legend()
    
    ls="-"
    if "+5" in dx_label:
        ls = "--"
    
    a_range, mean_Lx, std_Lx = averageModified(r[0],Lx_track[-1],dr_av = 0.1)

    if "a0=4" in dx_label:
        a_range = a_range[:-4]
        mean_Lx = mean_Lx[:-4]
        std_Lx = std_Lx[:-4]

    ax2.plot(a_range/l0, mean_Lx,ls=ls,label=f"{dx_label}",marker=".",color=f"C{k}")
    ax2.fill_between(a_range/l0, (mean_Lx-std_Lx*2)/np.max(np.abs(mean_Lx)), (mean_Lx+std_Lx*2)/np.max(np.abs(mean_Lx)),alpha=0.25,color=f"C{k}")
    mean_arr.append(mean_Lx)
    ax2.legend()
    plt.pause(0.01)
    k+=1

#/np.nanmax(np.abs(mean_Lx))

# fig1.tight_layout()
plt.axhline(0,ls="--",color="k")
fig2.tight_layout()
