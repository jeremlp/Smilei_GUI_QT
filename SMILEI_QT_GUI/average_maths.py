# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 15:54:51 2025

@author: Jeremy
"""

import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = f"{os.environ['SMILEI_SRC']}"
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
import time
from tqdm import tqdm
from smilei_utils import SET_CONSTANTS, averageAM, min_max,min_max_percentile
from scipy.signal import savgol_filter

plt.close("all")

l0 = 2*pi

a0_requested = 4.5
sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3.5_Tp12_dx128_AM4",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4.5_Tp12_dx128_AM4"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,3.5,4,4.5])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

# sim_path = "sim_NEW_TERM_a4_dx64_AM4"
# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a2_dx128_AM4_HD"

# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_HD"
# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_UHD"
# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a4_dx128_AM4_HD"
w0=2.5*l0
l1=1
def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)

zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)

N_part = 1
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ey","Ex"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = 1
track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()
if "AM" in sim_path:
    track_r_center = 0
else:
    track_r_center = Ltrans/2

y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
r = np.sqrt(y**2 + z**2)

RMIN, RMAX = 1*l0,2*l0

mask = np.where((r[0]>RMIN) & (r[0] < RMAX) )[0]


x = track_traj["x"][:,mask]
y = track_traj["y"][:,mask]-track_r_center
z = track_traj["z"][:,mask] -track_r_center
track_N = x.shape[1]

py = track_traj["py"][:,mask]
pz = track_traj["pz"][:,mask]
px = track_traj["px"][:,mask]
Ey = track_traj["Ey"][:,mask]
Ex = track_traj["Ex"][:,mask]

r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
# p_theta = p_theta = Lx_track/r
theta = np.arctan2(z,y)

x0 = x[0,0]

SET_CONSTANTS(w0, a0, Tp)



r0_requested = 1.55*l0
Nid = np.where(np.abs(r[0]-r0_requested) == np.min(np.abs(r[0]-r0_requested)))[0][0]
r0 = r[0,Nid]
theta0 = theta[0,Nid]
z0 = z[0,Nid]
y0 = y[0,Nid]

y_smooth = savgol_filter(y[:,Nid], window_length=35*5, polyorder=2)  # Window size should be odd
y_fast = y[:,Nid] - y_smooth

plt.figure()
plt.plot(t_range, y[:,Nid])
plt.plot(t_range, y_smooth,"--")
plt.plot(t_range,y_fast,"--")
plt.grid()
plt.title("Y(t)")

r_smooth = np.sqrt(y_smooth**2+z[:,Nid]**2)
r_fast = r[:,Nid] - r_smooth
plt.figure()
plt.plot(t_range, r[:,Nid])
plt.plot(t_range, r_smooth,"--")
plt.plot(t_range,r_fast,"--")
plt.grid()
plt.title("R(t)")

exact = integrate.simpson(r[:,Nid]/r_smooth* f(r[:,Nid],x0)**2*gauss(t_range,x0)**2, x=t_range)

rf2_smooth = savgol_filter(r_fast**2, window_length=35*5, polyorder=2)  # Window size should be odd
main =  integrate.simpson(f(r[:,Nid],x0)**2*gauss(t_range,x0)**2, x=t_range)
corr = integrate.simpson(rf2_smooth/(r_smooth)**2 * f(r[:,Nid],x0)**2*gauss(t_range,x0)**2, x=t_range)
print("(main-exact)/exact=",abs(main-exact)/exact*100,"%")
print(f"{exact:.3f}, {main+corr:.3f}, {corr:.2}")

plt.figure()
plt.plot(t_range,r_fast)
plt.grid()


g_smooth = savgol_filter(gamma[:,Nid], window_length=35*5, polyorder=2)  # Window size should be odd
g_fast = gamma[:,Nid] - g_smooth
g2_smooth = savgol_filter(g_fast**2, window_length=35*5, polyorder=2)  # Window size should be odd

plt.figure()
plt.plot(t_range, gamma[:,Nid])
plt.plot(t_range,g_smooth)
plt.plot(t_range,g_fast)
plt.plot(t_range,g_smooth+(g_smooth-1)*cos(t_range))
plt.xlabel("t")
plt.title("gamma(t)")
plt.grid()

exact = integrate.simpson(1/gamma[:,Nid] * gauss(t_range,x0)**2, x=t_range)
main =  integrate.simpson(1/g_smooth * gauss(t_range,x0)**2, x=t_range)
corr = integrate.simpson(g2_smooth/g_smooth**3 * gauss(t_range,x0)**2, x=t_range)
print("(main-exact)/exact=",abs(main-exact)/exact*100,"%")
print("(corr-exact)/exact=",abs(corr-exact)/exact*100,"%")
#0.3, 0.6, 0.8