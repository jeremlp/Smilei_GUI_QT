# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:14:27 2024

@author: Jeremy
"""
import numpy as np

import matplotlib.pyplot as plt
from numpy import pi, cos, sin, arctan2, exp, sqrt
import math
l0=2*pi


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy import integrate
from scipy import special

plt.close("all")

a0_requested = 2.33

sim_loc_list = ["SIM_OPTICAL_NR_HD/opt_base_PML_dx64",
                "SIM_OPTICAL_A0.3_HD/opt_a0.3_dx48",
                "SIM_OPTICAL_A1_HD/opt_a1.0_dx64",
                "SIM_OPTICAL_A1.5_HD/opt_a1.5_dx48",
                "SIM_OPTICAL_A2_HD/opt_a2.0_dx64",
                "SIM_OPTICAL_A2.33_HD/opt_a2.33_dx48",
                "SIM_OPTICAL_A2.5_HD/opt_a2.5_dx48",
                "SIM_OPTICAL_A3_HD/opt_a3.0_dx32"]
a0_range = np.array([0.1,0.3,1,1.5,2,2.33,2.5,3])

a0_sim_idx = np.where(a0_range==a0_requested)[0][0]
sim_path = sim_loc_list[a0_sim_idx]

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_A2_HD/opt_a2.0_dx64')


# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NR_HD/opt_base_PML_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt
import math
from numba import njit
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
x_pos = x[0,0]

tmax = 120

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

def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def f_prime(r,z):
    C_lp = np.sqrt(1/math.factorial(abs(l1)))
    return C_lp/w(z)**3 * exp(-(r/w(z))**2) * (r/w(z))**(abs(l1)-1) * (-2*r**2+w(z)**2*abs(l1))
def f_squared_prime(r,z):
    return 2*w0**2/(w(z)**2*r) * f(r,z)**2*(abs(l1)-2*(r/w0)**2+ 4*(z**2/w0**4))


def dr_long_time(r,t):
    pr_end = -a0**2/4*f_squared_prime(r, x_pos)*3/8*Tp
    return pr_end*(t-Tp-x_pos)

t_center=1.25*Tp
c_gauss = sqrt(pi/2)
sigma_gauss = Tp*3/8/c_gauss

def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)
def gauss2_int(t,x):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x_pos)/sigma_gauss)+1)


def pr_model_old(r):
    return -a0**2/4*f_squared_prime(r, x_pos)* 3/8*Tp
            # self.track_displace_pr_model, = self.ax4_displace.plot(r_range/l0, -self.a0**2/4*self.f_squared_prime(r_range,x_pos-self.xfoc)*3/8*self.Tp,"r--",label="Model")

def pr_model(r,t):
    return -a0**2/4*f_squared_prime(r, x_pos) * gauss2_int(t,x)

def pr_model_relat(r,t):
    return -1/gamma[:,Nid]*a0**2/4*f_squared_prime(r, x_pos)* gauss2_int(t,x)
def pr_model_relat_end(r):
    return -1/gamma[-1,Nid]*a0**2/4*f_squared_prime(r, x_pos)* gauss2_int(50*l0,x)


def EzModel(r,theta,z,t):
    phi = (w0**4 * (t - z)) / (w0**4 + 4 * z**2) + (2 * z * (-r**2 + 2 * (t - z) * z)) / (w0**4 + 4 * z**2)
    Omega = np.exp(-((r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0
    
    factor1 = 1 / (w0**4 + 4 * z**2)**5
    factor2 = 2 * np.sqrt(2) * np.exp(-((32 * np.pi * (t - 1.25 * Tp - z)**2) / (9 * Tp**2)))
    factor3 = w0**3 * Omega
    
    term1 = (2 * r**2 * z * (-12 * w0**2 * (w0**8 - 16 * z**4) + 
              r**2 * (5 * w0**8 - 40 * w0**4 * z**2 + 16 * z**4)) * 
              np.cos(2 * theta - phi))
    
    term2 = (2 * z * (2 * (3 * w0**4 - 4 * z**2) * (w0**4 + 4 * z**2)**2 - 
              16 * r**2 * w0**2 * (w0**8 - 16 * z**4) + 
              r**4 * (5 * w0**8 - 40 * w0**4 * z**2 + 16 * z**4)) * 
              np.cos(phi))
    
    term3 = (-r**2 * (r**2 * w0**2 * (w0**8 - 40 * w0**4 * z**2 + 80 * z**4) - 
              3 * (w0**12 - 20 * w0**8 * z**2 - 80 * w0**4 * z**4 + 64 * z**6)) * 
              np.sin(2 * theta - phi))
    
    term4 = ((2 * (w0**4 - 12 * z**2) * (w0**5 + 4 * w0 * z**2)**2 + 
              r**4 * w0**2 * (w0**8 - 40 * w0**4 * z**2 + 80 * z**4) - 
              4 * r**2 * (w0**12 - 20 * w0**8 * z**2 - 80 * w0**4 * z**4 + 64 * z**6)) * 
              np.sin(phi))
    
    return factor1 * factor2 * factor3 * (term1 + term2 + term3 + term4)


r_target=3.3*l0
Nid = np.where(np.abs(r[0]-r_target)==np.min(np.abs(r[0]-r_target)))[0][0]


plt.figure()
Y_test = np.cos(0.1*t_range)
plt.plot(t_range,Ex[:,Nid],".-",label="Ex")
plt.plot(t_range, 50*EzModel(r[:,Nid],theta[:,Nid],x[:,Nid],t_range),".-")
# plt.plot(t_range[:-3],Ex[3:,Nid],".-",label="Ex")

Ax = integrate.cumulative_trapezoid(Ex[:,Nid], x=t_range, initial=0)
# Ax2 = [integrate.simpson(Ex[:i,Nid],t_range[:i]) for i in range(1,len(Ex))]
# plt.plot(t_range,Ax,label="Ax")
# plt.plot(t_range[:-1],Ax2,"--",label="Ax2")

plt.grid()
plt.legend()

Int_Ex2 = integrate.simpson(Ex[:,Nid]**2,x=t_range)
Int_AxPx = integrate.simpson(px[:-3,Nid]*Ex[3:,Nid],x=t_range[:-3])
Int_ExPx = integrate.simpson(px[:,Nid]*Ex[:,Nid],x=t_range[:])
print(f"Ax using shift of {pi/(t_range[3]-t_range[0])}")
print("Ex^2 =",round(Int_Ex2,5))
print("Ex*px =",round(Int_ExPx,5))
print("Ax*px =",round(Int_AxPx,5))

