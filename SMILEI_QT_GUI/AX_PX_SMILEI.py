# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:06:13 2025

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
from numba import njit
from scipy import integrate
import scipy
from scipy.signal import savgol_filter
plt.close("all")

a0_requested = 3

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx48"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]


# sim_path = "SIM_PX_AX/gauss_a0.1_Tp12_PX_AX_dx64_AM8"

sim_path = "SIM_PX_AX/gauss_a3_Tp12_PX_AX_dx64_AM8"
# sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12"

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_A2_HD/opt_a2.0_dx64')

l0=2*np.pi
# T0 = S.TrackParticles("track_eon_exact", axes=["x","y","z","py","pz","px","Ex","Ey"])

try:
    T0 = S.TrackParticles("track_eon_exact", axes=["x","y","z","py","pz","px","Ex","Ey","Bz","Bx"])
except:
    T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Bz","Bx"])
# T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Bz","Bx"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1
dx = S.namelist.dx

track_N_tot = T0.nParticles
t_range = T0.getTimes()
t_range_smooth = np.arange(0,t_range[-1],0.01)
track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1

if "AM" in sim_path:
    dx_string = f"dx={l0/dx:.0f} AM={S.namelist.number_AM_modes}"
    track_r_center = 0
else:
    dx_string = f"dx={l0/dx:.0f}"
    track_r_center = Ltrans/2

x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

Ey = track_traj["Ey"][:,::N_part]
Ex = track_traj["Ex"][:,::N_part]
Bz = track_traj["Bz"][:,::N_part]
Bx = track_traj["Bx"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
vx = px/gamma

def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)

def gauss(tau):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((tau-t_center)/(Tp*3/8/c_gauss))**2)

def get_EB_fast(x,y,z,t):
    l,eps = 1,0
    zR = 0.5*w0**2
    C_lp = sqrt(2)
    g = gauss
    
    r = sqrt(x**2+y**2)
    theta = arctan2(y,x)
    tau = t-z
    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l)+1)*arctan2(z,zR) + pi
    a_ = a0*w0/w_/sqrt(1+abs(eps)) 
    f_ = C_lp * (r/w_)**abs(l)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l)-1) * (-2*r**2+w_**2*abs(l))
    
    Ey = -a_*f_*g(tau)*cos(l*theta + phi_)
    Ez = -eps*a_*f_*g(tau)*sin(l*theta + phi_)
    Ex = a_*(f_prime_*g(tau)*cos(theta))*sin(l*theta + phi_) -\
        a_*f_*g(tau)*(l/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l*theta+phi_)
        
    By = eps*a_*f_*g(tau)*sin(l*theta + phi_)
    Bz = -a_*f_*g(tau)*cos(l*theta + phi_)
    Bx = a_*(f_prime_*g(tau)*sin(theta))*sin(-l*theta-phi_) +\
            a_*f_*g(tau)*(l/r*cos(theta)+r*z/(zR*w_**2)*sin(theta))*cos(-l*theta-phi_)
    
    return Ex, Ey, Ez,  Bx, By, Bz


r0_requested = 1*w0/sqrt(2)

Nid = np.where(np.abs(r[0]-r0_requested)==np.min(np.abs(r[0]-r0_requested)))[0][0]-2
# Nid = 2
r0 = r[0,Nid]
x0 = x[0,Nid]

print("INIT POS:",x0/l0,r0/l0)

ExM, EyM, EzM, BxM, ByM, BzM = get_EB_fast(y[:,Nid], z[:,Nid], x[:,Nid],t_range)
AxM, AyM, AzM, _, _, _ = get_EB_fast(y[:,Nid], z[:,Nid], x[:,Nid],t_range+pi/2)
window_length = 201
if len(py) > 1500: window_length = 2001
py_slow = savgol_filter(py[:,Nid], window_length=window_length,polyorder=2)
py_fast = py[:,Nid]-py_slow

px_slow = savgol_filter(px[:,Nid], window_length=window_length,polyorder=2)
px_fast = px[:,Nid]-px_slow

AxMS = -integrate.cumulative_trapezoid(Ex[:,Nid], t_range, initial=0)
AxMS_slow = savgol_filter(AxMS, window_length=window_length,polyorder=2)
AxMS_fast = AxMS-AxMS_slow

AyMS = -integrate.cumulative_trapezoid(Ey[:,Nid], t_range, initial=0)
AyMS_slow = savgol_filter(AyMS, window_length=window_length,polyorder=2)
AyMS_fast = AyMS-AyMS_slow

plt.figure()
# plt.plot(t_range/l0, Ey, label="Ey")
plt.plot(t_range/l0, AyM, label="Ay")
plt.plot(t_range/l0, py_fast, label="py")
plt.grid()
plt.xlabel("t/t0")
plt.xlim(14,29)
plt.legend()


f0 = f(r0,x0)
COEF = sqrt(1+(f0*a0)**2+1/4*(f0*a0)**4)
print(f"{COEF=}")
plt.figure()
# plt.plot(t_range/l0, Ex, label="Ex")
plt.plot(t_range/l0, COEF*AxM, label="$\gamma$*Ax")
plt.plot(t_range/l0, px_fast, label="px_f")
plt.grid()
plt.xlabel("t/t0")
plt.xlim(14,29)
plt.legend()


plt.figure()
# plt.plot(t_range/l0, Ex, label="Ex")
plt.plot(t_range/l0, ExM, label="ExM")
plt.plot(t_range/l0, Ex[:,Nid], label="Ex")
plt.grid()
plt.xlabel("t/t0")
plt.xlim(12,31)
plt.legend()

# plt.figure()
# # plt.plot(t_range/l0, Ex, label="Ex")
# plt.plot(t_range/l0, EyM, label="EyM")
# plt.plot(t_range/l0, Ey[:,Nid], label="Ey")
# plt.grid()
# plt.xlabel("t/t0")
# plt.xlim(12,31)
# plt.legend()
# plt.figure()
# # plt.plot(t_range/l0, Ex, label="Ex")
# plt.plot(t_range/l0, AxM*AxM, label="Ax^2")
# plt.plot(t_range/l0, px_fast*AxM, label="px_f*Ax")
# plt.grid()
# plt.xlabel("t/t0")
# plt.xlim(14,29)
# plt.legend()

print("int px*AxM / int AxM^2=",integrate.simpson(px_fast*AxM)/integrate.simpson(AxM**2))
print("int px*AxMS / int AxMS^2=",integrate.simpson(px_fast*AxMS)/integrate.simpson(AxMS**2))
print("int px*AxMS_f / int AxMS_f^2=",integrate.simpson(px_fast*AxMS_fast)/integrate.simpson(AxMS_fast**2))


px_Ax_mean = savgol_filter(px_fast*AxMS_fast, window_length=window_length*2,polyorder=3)
Ax2_mean = savgol_filter(AxMS_fast**2, window_length=window_length*2,polyorder=3)

plt.figure()
plt.plot(t_range/l0,px_Ax_mean,label="<$p_x\cdot A_x$>")
plt.plot(t_range/l0,Ax2_mean,"--",label="<$ A_x^2$>")
plt.plot(t_range/l0,Ax2_mean*sqrt(1+(f0*a0)**2+1/4*(f0*a0)**4),"--",label="$\gamma_{max}< A_x^2 >$")
plt.plot(t_range/l0,Ax2_mean*sqrt(1+(f0*a0)**2/2 + 1/16*(f0*a0)**4),"--",label="$<\gamma><A_x^2>$")
plt.grid()
plt.xlabel("t/t0")
plt.legend()


gamma_mean = savgol_filter(gamma[:,Nid], window_length=window_length*2,polyorder=2)
plt.figure()
plt.plot(t_range/l0,gamma[:,Nid],label="$\gamma(t)$")
plt.plot(t_range/l0, gamma_mean,label="<$ \gamma(t) $>")
plt.plot(t_range/l0, COEF+t_range*0,"--",label="$\gamma_{max}$")
# COEF = sqrt(1+(f0*a0)**2+1/4*(f0*a0)**4)
plt.plot(t_range/l0, sqrt(1+(f0*a0)**2/2+1/16*(f0*a0)**4)+t_range*0,"--",label="$\gamma_{mean}$")

plt.grid()
plt.legend()
plt.xlabel("t/t0")




#==========================
# COMPUTE THE AVERAGES OF OTHER INT TERMS
#==========================


g = gamma[:,Nid]

plt.figure()
int_g = integrate.cumulative_trapezoid(px_fast**2/2* np.gradient(1/g,t_range),x=t_range)
plt.plot(t_range[:-1]/l0,g[:-1]*int_g*AxMS[:-1],label="g*int_psi *Ax")
plt.grid()
plt.xlabel("t/t0")
plt.legend()
plt.axhline(0,ls="--",color="k")
print(integrate.simpson(g[:-1]*int_g*AxMS_fast[:-1],x=t_range[:-1]))


plt.figure()
int_Ay = integrate.cumulative_trapezoid(1/(2*g)* np.gradient(AyMS_fast**2,t_range),x=t_range)
plt.plot(t_range[:-1]/l0,g[:-1]*int_Ay*AxMS_fast[:-1],label="g*int_Ay2 *Ax")
plt.grid()
plt.xlabel("t/t0")
plt.legend()
plt.axhline(0,ls="--",color="k")
print(integrate.simpson(g[:-1]*int_Ay*AxMS_fast[:-1],x=t_range[:-1]))



plt.figure()
int_AyAx = integrate.cumulative_trapezoid(AyMS_fast/(g)* np.gradient(AxMS_fast,t_range),x=t_range)
plt.plot(t_range[:-1]/l0,g[:-1]*int_AyAx*AxMS_fast[:-1],label="g*int_AyAx *Ax")
plt.grid()
plt.xlabel("t/t0")
plt.legend()
print(integrate.simpson(g[:-1]*int_AyAx*AxMS_fast[:-1],x=t_range[:-1]))
plt.axhline(0,ls="--",color="k")

plt.figure()
plt.plot(t_range[:-1]/l0,g[:-1]*AxMS_fast[:-1]*AxMS_fast[:-1],label="Ax^2")
plt.grid()
plt.xlabel("t/t0")
plt.legend()
plt.axhline(0,ls="--",color="k")
print(integrate.simpson(AxMS_fast[:-1]**2,x=t_range[:-1]))







x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

Ey = track_traj["Ey"][:,::N_part]
Ex = track_traj["Ex"][:,::N_part]
Bz = track_traj["Bz"][:,::N_part]
Bx = track_traj["Bx"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)

vx = px/gamma
vr = pr/gamma


Br = z*Bz/r #(y*By + z*Bz)/r

Etheta = -z*Ey/r #(y*Ez - z*Ey)/r


Lorentz = -Etheta - (vx*Br -vr*Bx)

plt.figure()
plt.plot(t_range[:-1],integrate.cumulative_simpson((r*Lorentz)[:,-2],x=t_range))
plt.plot(t_range,Lx_track[:,-2],"--")
plt.grid()










