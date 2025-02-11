# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:12:29 2025

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
import time
from tqdm import tqdm
plt.close("all")

a0_requested = 2

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

# sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6"
# sim_path = "SIM_OPTICAL_A2_HD/opt_a2.0_dx64"

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp6')

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/gauss_a2_Tp6_NET_GAIN_dx64')

l0=2*np.pi
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])

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
if "AM" in sim_path:
    track_r_center = 0
else:
    track_r_center = Ltrans/2
x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part] - track_r_center
z = track_traj["z"][:,::N_part] - track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
vx = px/gamma
x0 = x[0,0]



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

def LxEpolar_V2_O3(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 +
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) -
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 +
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) +
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr
def LxEpolar_V2_O3(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O3(r,theta,z)

def LxEpolar_V2_O5(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O5(r,theta,z)

def Ftheta_V2_O3(r,theta,z):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3

# @njit
def Torque_V2_O3(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r**2 * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3

# @njit
def Ftheta_V2_O5(r,theta,z):
    numerator = (
        2 * a0**2 * r * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (
            2 * z * np.cos(2 * theta) * (
                4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)
            ) +
            np.sin(2 * theta) * (
                4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * (
                    8 * (w0**4 + 4 * w0**2 + 20) * z**2 +
                    (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4
                ) -
                (w0**4 + 4 * z**2) * (
                    4 * (w0**2 - 2) * z**2 +
                    (w0**2 + 4) * (w0**2 + 6) * w0**2
                )
            )
        )
    )
    denominator = (w0**4 + 4 * z**2)**5
    
    expression = numerator / denominator
    return expression
def Torque_V2_O5(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    numerator = ( 2 * a0**2 * r**2 * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (2 * z * np.cos(2 * theta) * ( 4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)) +
            np.sin(2 * theta) * (  4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * ( 8 * (w0**4 + 4 * w0**2 + 20) * z**2 + (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4) -
                (w0**4 + 4 * z**2) * ( 4 * (w0**2 - 2) * z**2 + (w0**2 + 4) * (w0**2 + 6) * w0**2  ))))
    denominator = (w0**4 + 4 * z**2)**5
    return numerator / denominator


def gauss_squared_prime(t,x):
    t_center = 1.25*Tp
    c_gauss = sqrt(pi/2)
    sigma = 3/8*Tp/c_gauss
    return 4/sigma**2 * (t-t_center-x) * gauss(t,x)**2

zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)

def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)
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

t_center=1.25*Tp
c_gauss = sqrt(pi/2)
sigma_gauss = Tp*3/8/c_gauss


def gauss2_int(t,x):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x0)/sigma_gauss)+1)
def gauss2_int_int(t,x):
    t_center = 1.25*Tp
    psi = lambda t,x : sqrt(2)*(t-t_center-x)/sigma_gauss
    expr =lambda t,x : 0.5*sqrt(pi/2)*sigma_gauss*( gauss(t,x)**2*sigma_gauss/sqrt(2*pi) + t + (t-t_center-x)*(special.erf(psi(t,x))))
    return expr(t,x) - expr(0,x)

def dr_Relat_mean(r,t,x):
    return -1/(1+(f(r,x)*a0)**2/2+ 1/4*1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_Relat(r,t,x):
    return -1/(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr(r,t,x):
    return -1/sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

#==================================================
r_requested = 0.5*l0
Nid = np.where((np.abs(r[0]-r_requested)<0.001*l0) & (np.abs(Lx_track[-1])>0.021/10))[0][0]
Interaction_Id = np.where(gauss(t_range,x[:,Nid])>0.2)[0]
#==================================================


x0 = x[0,Nid]
r0 = r[0,Nid]
theta0 = theta[0,Nid]
y_range = np.arange(-w0,w0+0.1,0.1)

Y,Z = np.meshgrid(y_range,y_range)
R = np.sqrt(Y**2+Z**2)
THETA = np.arctan2(Z,Y)

extent = [y_range[0]/l0, y_range[-1]/l0, y_range[0]/l0, y_range[-1]/l0]

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(14,5))

torque_2D = Torque_V2_O5(R, THETA,x0)

r_model = np.abs(r0+ dr(r0,t_range,x0))

im = ax1.imshow(torque_2D, cmap="RdBu",extent=extent,aspect="auto",label="Torque Model",origin="lower")
ax1.plot(y[Interaction_Id,Nid]/l0, z[Interaction_Id,Nid]/l0, "k",label="electron trajectory Smilei")
ax1.plot((r0+ dr(r0,t_range,x0))[Interaction_Id]/l0*np.cos(theta[0,Nid]), (r0+ dr(r0,t_range,x0))[Interaction_Id]/l0*np.sin(theta[0,Nid]), 
         "C1--",label="electron trajectory Model")

ax1.scatter(y[0,Nid]/l0, z[0,Nid]/l0, color="r",s=250, marker="x",label="initial position")
fig.colorbar(im,ax=ax1,pad=0.01)
ax1.set_xlabel("$y/\lambda$")
ax1.set_ylabel("$z/\lambda$")
ax1.set_title("Electron motion inside the torque field \nfrom the model and from simulation")

ax1.legend()
ax1.set_xlim(-w0/l0,w0/l0)
ax1.set_ylim(-w0/l0,w0/l0)

ax2.plot(t_range/l0, r[:,Nid]/l0,"-",label="Smilei $r(t)$")
ax2.plot(t_range/l0, r_model/l0, "C1",label="Model $r_0+\Delta r(t)$")
ax2.set_title("Radial coordinates $r(t)/\lambda$")

ax2.grid()
ax2.set_xlabel("$t~c/\lambda$")
ax2.set_ylabel("$r/\lambda$")
ax2.set_xlim(13,26)
ax2.set_ylim(-0.5/l0,7/l0)

ax2.legend()

ax3.plot(t_range/l0, Torque_V2_O5(r[:,Nid],theta[:,Nid],x[:,Nid])*gauss(t_range,x[:,Nid])**2,"-",label="Torque with Smilei $r(t)$")
ax3.plot(t_range/l0, Torque_V2_O5(r_model,theta0,x[:,Nid])*gauss(t_range,x[:,Nid])**2, "C1",label="Torque with Model $r_0+\Delta r(t)$")
ax3.grid()
ax3.set_xlabel("$t~c/\lambda$")
ax3.set_ylabel("Torque")
ax3.set_xlim(13,26)
ax3.set_title("Torque computed using \nModel or Smilei radial coordinates")

ax3.legend()

fig.tight_layout()