# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:36:13 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
S = happi.Open('C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_PHYSICAL/sim_SAM_Long')
l0=2*np.pi


diag_px = S.ParticleBinning("px")
diag_py = S.ParticleBinning("py")
diag_pz = S.ParticleBinning("pz")

px_range = diag_px.getAxis("px")
t_range = diag_px.getTimes()/l0

px = np.array(diag_px.getData())
py = np.array(diag_py.getData())
pz = np.array(diag_pz.getData())

time_idx = -1

plt.plot(px_range, px[time_idx],label='px')
plt.plot(px_range, py[time_idx],label='py')
plt.plot(px_range, pz[time_idx],label='pz')
plt.grid()
plt.legend()
plt.xlabel("p_i")
plt.ylabel("weight")
plt.title("sim_OAM_cut (t/t0=160)")
plt.tight_layout()


def std_distrib(a,weight):
    px_range_r = np.repeat(a[None,:],160,axis=0)
    weighted_mean = np.average(px_range_r, weights=weight,axis=1)
    weighted_variance = np.average((px_range_r.T - weighted_mean.T).T ** 2, weights=weight,axis=1)
    std_px = np.sqrt(weighted_variance)
    return weighted_mean,std_px


p0_x, pt_x = std_distrib(px_range,px)
p0_y, pt_y = std_distrib(px_range,py)
p0_z, pt_z = std_distrib(px_range,pz)



plt.figure()
plt.plot(t_range, pt_x, label="pt_x")
plt.plot(t_range, pt_y, label="pt_y")
plt.plot(t_range, pt_z, label="pt_z")
plt.plot(t_range, np.abs(p0_x), "k-",label="|p0|")
plt.legend()
plt.grid()
plt.title("sim_OAM_cut")
plt.xlabel("t/t0")


plt.tight_layout()

# print("p0:",round(p0,3))
# print("weibel cond",round(pt_z,4), round(np.sqrt(1+p0**2)*p0,4))
