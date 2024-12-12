# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:25:00 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = r'C:\Users\Jeremy\_LULI_\Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy import integrate
plt.close("all")

requested_a0 = 2
if requested_a0==2:
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/opt_a2.0_dx64')
else:
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/gauss_a2_Tp6')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NR_HD/opt_base_PML_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt
import math
from numba import njit
import time

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

simu_list = ['gauss_a2_Tp6',
             'gauss_a2_Tp12_dx32',
             'gauss_a2_Tp12']

# simu_list = ['gauss_a2_Tp6',
#              'gauss_a2_Tp12']
#             "nff_SuperGaussian_a2_Tp6_w2.5"
fig1,ax1 = plt.subplots()

for k,simu in enumerate(simu_list):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/{simu}')
    T0 = S.TrackParticles("track_eon_full", axes=["x","y","z","py","pz","px"])

    Ltrans = S.namelist.Ltrans
    Tp = S.namelist.Tp
    w0 = S.namelist.w0
    a0 = S.namelist.a0
    l1 = S.namelist.l1


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
    theta = np.arctan2(z,y)
    pr = (y*py + z*pz)/r
    Lx_track =  y*pz - z*py
    gamma = sqrt(1+px**2+py**2+pz**2)


    t_min, t_max = 0,40
    t_min_idx = np.where(np.abs(t_range/l0-t_min)==np.min(np.abs(t_range/l0-t_min)))[0][0]
    t_max_idx = np.where(np.abs(t_range/l0-t_max)==np.min(np.abs(t_range/l0-t_max)))[0][0]
    a_range,m,M = min_max(r[0], Lx_track[-1])
    # ax1.plot(a_range/l0,m,f"C{k}",lw=2)
    # ax1.plot(a_range/l0,M,f"C{k}",lw=2,label=f"{simu}(dx={l0/S.namelist.dx:.0f})\nt={t_range[-1]/l0:.2f}/t0")

    var_list = []

    a_range, av_Lx, std_Lx = averageAM(r[0], Lx_track[-1],1)
    ax1.plot(a_range/l0,av_Lx,f"C{k}-",label=f"{simu} (dx={l0/S.namelist.dx:.0f})")
    ax1.fill_between(a_range/l0,av_Lx+std_Lx,av_Lx-std_Lx,alpha=0.2,color=f"C{k}")

ax1.grid()
ax1.set_xlabel("$r_0/\lambda$")
ax1.set_ylabel("<Lx>")
ax1.legend()
fig1.suptitle("Averaged <Lx(r)> for different simulations")
fig1.tight_layout()








