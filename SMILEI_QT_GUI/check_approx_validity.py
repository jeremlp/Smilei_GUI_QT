# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:33:44 2025

@author: Jeremy
"""

import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from numba import njit
import time
from tqdm import tqdm
plt.close("all")

l0=2*pi

def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def gauss(tau):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((tau-t_center)/(Tp*3/8/c_gauss))**2)

w0 = 2.5*l0
zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)
g = gauss
def getE(r,theta,z,t):
    tau = t-z
    
    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l1)+1)*arctan2(z,zR)
    a_ = a0*w0/w_/sqrt(1+abs(eps))
    f_ = C_lp * (r/w_)**abs(l1)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l1)-1) * (-2*r**2+w_**2*abs(l1))
    
    Ex = a_*f_*g(tau)*cos(l1*theta + phi_)
    Ey = -eps*a_*f_*g(tau)*sin(l1*theta + phi_)
    Ez = -a_*(f_prime_*g(tau)*cos(theta))*sin(l1*theta + phi_) +\
        a_*f_*g(tau)*(l1/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l1*theta+phi_)
    return Ex,Ey,Ez



a0_requested = 0.1

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4.5_Tp12_dx48"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4,4.5])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

# sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6"
# sim_path = "SIM_OPTICAL_A2_HD/opt_a2.0_dx64"

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12')

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/gauss_a2_Tp6_NET_GAIN_dx64')

l0=2*np.pi
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Bx","Bz"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1


track_N_tot = T0.nParticles
t_range = T0.getTimes()
t_range_smooth = np.arange(0,t_range[-1],0.5)
track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1
# r_range_net = S.namelist.r_range_net

x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

Ex = track_traj["Ex"][:,::N_part]
Ey = track_traj["Ey"][:,::N_part]
Bx = track_traj["Bx"][:,::N_part]
Bz = track_traj["Bz"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
ptheta = (y*pz - z*py)/r
p_perp = sqrt(pr**2 + ptheta**2)

Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)

vx = px/gamma

plt.figure()
plane_wave_approx_error = np.nanmax([(1-np.nanmin(gamma-px))*100,(np.nanmax(gamma-px)-1)*100])

plt.title(f"$\gamma$ - px = 1 for {a0=} (Error= {plane_wave_approx_error:.2f}%)")
plt.plot(t_range/l0,gamma-px,alpha=0.5)
plt.plot(t_range/l0,np.mean(gamma-px,axis=-1),lw=3,color="r")
plt.grid()
plt.axhline(1,color="k", ls="--")
plt.xlabel("t/t0")
plt.ylabel("gamma-px")
# print(np.nanmin(gamma-px),np.nanmax(gamma-px))
print("gamma-px error:",plane_wave_approx_error,"%") 


r0_requested = w0/sqrt(2)
Nid = np.where( np.abs(r[0]-r0_requested)==np.min(np.abs(r[0]-r0_requested)))[0][0]
x0 = x[0,Nid]
r0 = r[0,Nid]
print("r0=",r0/l0)

AyM, AzM, AxM = getE(r[:,Nid],theta[:,Nid],x[:,Nid],t_range+pi/2)
EyM, EzM, ExM = getE(r[:,Nid],theta[:,Nid],x[:,Nid],t_range)

COEF = np.sqrt(1 + (f(r0,x0)*a0)**2 + 1/4*(f(r0,x0)*a0)**4)

plt.figure()

plt.plot(t_range/l0,py[:,Nid],label="py")
plt.plot(t_range/l0,AyM,"-",label="AyM")

smooth_p = savgol_filter(py[:,Nid], window_length=71,  polyorder=2)
plt.plot(t_range/l0,py[:,Nid]-smooth_p,"--",label="py fast")
# plt.plot(t_range/l0,Ex[:,Nid],"--",label="Ex")
# plt.plot(t_range/l0,Bx[:,Nid],"--",label="Bx")
plt.title("Perp")
plt.xlabel("t/t0")
plt.grid()
plt.legend()

plt.figure()
plt.title("Longitudinal")
# plt.plot(t_range/l0,px[:,Nid],label="px")
plt.plot(t_range/l0,AxM,label="AxM")
plt.plot(t_range/l0,gamma[:,Nid]*AxM,label="$\gamma$*AxM")
plt.plot(t_range/l0,COEF*AxM,label="$\gamma_{max}$*AxM")

smooth_p = savgol_filter(px[:,Nid], window_length=71,  polyorder=2)
plt.plot(t_range/l0,px[:,Nid]-smooth_p,"--",label="px fast")
# plt.plot(t_range/l0, Bx[:,Nid],label="Bx")
plt.xlabel("t/t0")
plt.grid()
plt.legend()

plt.figure()
smooth_A2 = savgol_filter(AxM**2, window_length=101,  polyorder=2)

smooth_Ap = savgol_filter(px[:,Nid]*AxM, window_length=101,  polyorder=2)

smooth_A2_gamma2 = savgol_filter(-gamma[:,Nid]*AxM**2, window_length=101,  polyorder=2)

smooth_A2_gamma_m = savgol_filter(sqrt(1+(f(r0,5*l0)*a0)**2 + 0.25*(f(r0,5*l0)*a0)**4)*AxM**2, window_length=101,  polyorder=2)
smooth_A2_gamma_m2 = savgol_filter(sqrt(1+(f(r0,5*l0)*a0)**2 + (f(r0,5*l0)*a0)**4)*AxM**2, window_length=101,  polyorder=2)


plt.plot(t_range/l0,smooth_A2,"k--",label="<Ax^2>")
plt.plot(t_range/l0,smooth_Ap,label="<Ax*px>")
# plt.plot(t_range/l0,smooth_A2_gamma2,label="<$\gamma$*Ax^2>")
plt.plot(t_range/l0,smooth_A2_gamma_m,label="<$\gamma_{max,1/4}$*Ax^2>")
plt.plot(t_range/l0,smooth_A2_gamma_m2,label="<$\gamma_{max,1}$*Ax^2>")

plt.grid()
plt.xlabel("t/t0")
plt.title(f"Ax*px vs Ax^2 for {a0=}")
plt.legend()

print(integrate.simpson(smooth_Ap))
print(integrate.simpson(smooth_A2_gamma_m))
print(integrate.simpson(smooth_A2_gamma_m2))