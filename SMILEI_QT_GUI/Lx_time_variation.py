# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:25:00 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = r'C:\Users\jerem\Smilei'
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

simu_list = ["nff_sin2_a2_Tp6_w2.5",
             'gauss_a2_Tp6',
             'gauss_a2_Tp6_dx48',
             'gauss_a2_Tp12']
#             "nff_SuperGaussian_a2_Tp6_w2.5"
fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()


for k,simu in enumerate(simu_list):
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{simu}')
    T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","Bx","By","Bz"])

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

    Ey = track_traj["Ey"][:,::N_part]
    Ez = track_traj["Ez"][:,::N_part]
    Ex = track_traj["Ex"][:,::N_part]
    By = track_traj["By"][:,::N_part]
    Bz = track_traj["Bz"][:,::N_part]
    Bx = track_traj["Bx"][:,::N_part]

    r = np.sqrt(y**2+z**2)
    theta = np.arctan2(z,y)
    pr = (y*py + z*pz)/r
    Lx_track =  y*pz - z*py
    gamma = sqrt(1+px**2+py**2+pz**2)


    t_min, t_max = 0,40
    t_min_idx = np.where(np.abs(t_range/l0-t_min)==np.min(np.abs(t_range/l0-t_min)))[0][0]
    t_max_idx = np.where(np.abs(t_range/l0-t_max)==np.min(np.abs(t_range/l0-t_max)))[0][0]
    a_range,m,M = min_max(r[0], Lx_track[-1])
    ax1.plot(a_range/l0,m,f"C{k}",lw=2)
    ax1.plot(a_range/l0,M,f"C{k}",lw=2,label=f"{simu}(dx={l0/S.namelist.dx:.0f})\nt={t_range[-1]/l0:.2f}/t0")
    # for i,t in enumerate(range(t_min_idx,t_max_idx,2)):

    #     a_range,m,M = min_max(r[0], Lx_track[t])
    #     plt.plot(a_range/l0,m,f"C{i}")
    #     plt.plot(a_range/l0,M,f"C{i}",label=f"t={t_range[t]/l0:.2f}/t0")



    # a_range,m,M = min_max(r[0], Lx_track[-1])
    # plt.plot(a_range/l0,m,"k",lw=2)
    # plt.plot(a_range/l0,M,"k",lw=2,label=f"t={t_range[-1]/l0:.2f}/t0")

    # plt.xlabel("$r_0/\lambda$")
    # plt.legend()
    # plt.grid()

    # plt.figure()
    var_list = []
    for i,t in enumerate(range(t_min_idx,t_max_idx,1)):
        a_range,m,M = min_max(r[0], Lx_track[t])
        a_range_var_idx = np.where(np.abs(a_range/l0-3)==np.min(np.abs(a_range/l0-3)))[0][0]


        var_list.append(m[a_range_var_idx])

    ax2.plot(t_range[t_min_idx:t_max_idx]/l0,var_list,".-",label=f"{simu} (dx={l0/S.namelist.dx:.0f})")
    ax3.plot(t_range[t_min_idx:t_max_idx]/l0,np.abs(np.gradient(var_list,t_range[t_min_idx:t_max_idx])),".-",label=f"{simu} (dx={l0/S.namelist.dx:.0f})")

ax1.grid()
ax1.set_xlabel("$r_0/\lambda$")
ax1.set_ylabel("Lx")
ax1.legend()
fig1.suptitle("Lx(r) for different simulations")
fig1.tight_layout()


ax2.grid()
ax2.set_xlabel("$t/t0$")
ax2.set_ylabel("Lx at $r=3\lambda$")
ax2.legend()
fig2.suptitle("Time variation of Lx at $r=3\lambda$")
fig2.tight_layout()

ax3.grid()
ax3.set_xlabel("$t/t0$")
ax3.set_ylabel("dLx/dt at $r=3\lambda$")
ax3.set_yscale("log")
X0=0.03
ax3.axhline(0.1*1/20*X0/l0,color="k",ls="--",label="10% <Lx> variation")
ax3.legend()
fig3.suptitle("Time derivative of Lx at $r=3\lambda$, in log scale")
fig3.tight_layout()






