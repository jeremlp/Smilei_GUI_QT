# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:02:45 2025

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
    


sim_loc_list_12 = ["gauss_a4_Tp6_NET_GAIN_dx128_AM4",
                    "gauss_a4_Tp9_NET_GAIN_dx128_AM4",
                    "gauss_a4_Tp10_NET_GAIN_dx128_AM4",
                    "gauss_a4_Tp12_NET_GAIN_dx128_AM4",
                    "gauss_a4_Tp14_NET_GAIN_dx128_AM4",
                    "gauss_a4_Tp16_NET_GAIN_dx128_AM4"]


Tp_range = np.array([6,9,10,12,14,16])


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
    

    

mean_arr = []
k=0
for sim in sim_loc_list:
    
    try: 
        data = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/net_gain_smilei/net_gain_{sim}.txt")
        a_range, mean_Lx, std_Lx = data[:,0], data[:,1], data[:,2]
    except:
    
        S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/{sim}')
        
        T0 = S.TrackParticles("track_eon_net", axes=["x","y","z","py","pz","px"])
        
        Ltrans = S.namelist.Ltrans
        a0 = S.namelist.a0
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
        
        a_range, mean_Lx, std_Lx = averageModified(r[0],Lx_track[-1],dr_av = 0.1)
        np.savetxt(f"{os.environ['SMILEI_QT']}/data/net_gain_smilei/net_gain_{sim}.txt",np.column_stack((a_range, mean_Lx,std_Lx)))

    

    # if Tp_range > 4:
    #     a_range = a_range[:-4]
    #     mean_Lx = mean_Lx[:-4]
    #     std_Lx = std_Lx[:-4]
    
    
    mean_arr.append(np.max(np.abs(mean_Lx)))
    k+=1

fig, ax = plt.subplots(1)

plt.loglog(Tp_range, mean_arr,"o",ms=12)

ax.set_xlabel("$T_p$", fontsize=16)
ax.grid()
ax.set_title(f"Average $<L_x>$ (normalized) radial distribution\n($a_0={a0},w_0=2.5\lambda$)")
# ax2.set_title(f"Mean <Lx> (normalized) radial distribution\n($Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
ax.set_ylabel("$<L_x>$", fontsize=16)

a,b = np.polyfit(np.log10(Tp_range)[1:-2], np.log10(mean_arr)[1:-2],1)
print(a,b)
plt.loglog(Tp_range, 10**(a*np.log10(Tp_range)+b),"--",label=f"fit {a:.2f}")
plt.legend()
fig.tight_layout()
plt.pause(0.1)
plt.axhline(0,ls="--",color="k")
fig.tight_layout()





plt.figure()
for Tp in [6,9,10,12,14]:
    sim = f"gauss_a4_Tp{Tp}_NET_GAIN_dx128_AM4"
    data = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/net_gain_smilei/net_gain_{sim}.txt")
    a_range, mean_Lx, std_Lx = data[:,0], data[:,1], data[:,2]
    plt.plot(a_range/l0,mean_Lx,".-",label=f"Tp={Tp}$\lambda/c$")

plt.grid()
plt.legend()